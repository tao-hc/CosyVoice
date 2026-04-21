# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

import threading
import time
import uuid
import vllm
import itertools
import queue
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import datetime
import multiprocessing
import json
import gc
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.frontend import CosyVoiceFrontEnd

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

print("[INFO] Initializing NPU environment...")
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    # 设置编译模式
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    print("[INFO] NPU initialized successfully.")
except ImportError:
    print("[WARNING] NPU libraries not found. Using CPU.")



def process_prompt_wave_info(speaker_info_dir, frontend, sample_rate):
    speaker_info_file = os.path.join(speaker_info_dir, 'speaker_info.json')
    with open(speaker_info_file, 'r', encoding='utf-8') as f:
        all_speakers = json.load(f)

    prompt_wave_info = {}
    for spk_id, wave_info in all_speakers.items():
        prompt_text = wave_info['prompt_text']
        prompt_wav = load_wav(os.path.join(speaker_info_dir, wave_info['prompt_wav']), 16000)
        model_input = frontend.frontend_zero_shot('', prompt_text, prompt_wav, sample_rate, '')

        prompt_wave_info[spk_id] = {
            'prompt_wav': prompt_wav,
            'prompt_text': wave_info['prompt_text'],
            'prompt_text': model_input['prompt_text'],
            'prompt_text_len': model_input['prompt_text_len'],
            'llm_prompt_speech_token': model_input['llm_prompt_speech_token'],
            'llm_prompt_speech_token_len': model_input['llm_prompt_speech_token_len'],
            'flow_prompt_speech_token': model_input['flow_prompt_speech_token'],
            'flow_prompt_speech_token_len': model_input['flow_prompt_speech_token_len'],
            'prompt_speech_feat': model_input['prompt_speech_feat'],
            'prompt_speech_feat_len': model_input['prompt_speech_feat_len'],
            'llm_embedding': model_input['llm_embedding'],
            'flow_embedding': model_input['flow_embedding'],
        }

    return prompt_wave_info

class Flow_And_Hift():
    def __init__(self,
                 device,
                 fp16: bool = False,
                 model_dir: str = None,
                 speaker_info_dir: str = None,
                 graph_mode: bool = False):
        hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        self.device = device
        self.flow = configs['flow'].to(self.device)
        self.hift = configs['hift'].to(self.device)
        self.voice_sample_rate = configs['sample_rate']
        self.token_sample_rate = configs['sample_rate'] / 960

        self.model_dir = model_dir
        self.graph_mode = graph_mode

        frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                     configs['feat_extractor'],
                                     os.path.join(model_dir, 'campplus.onnx'),
                                     os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
                                     os.path.join(model_dir, 'spk2info.pt'),
                                     configs['allowed_special'])
        sample_rate = configs['sample_rate']

        flow_ckpt_file = os.path.join(model_dir, 'flow.pt')
        hift_ckpt_file = os.path.join(model_dir, 'hift.pt')

        self.flow.load_state_dict(torch.load(flow_ckpt_file, map_location=self.device, weights_only=True), strict=True)
        self.flow.eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_ckpt_file, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.eval()

        self.prompt_wave_info = process_prompt_wave_info(speaker_info_dir, frontend, sample_rate)

        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        self.split_token_len = 10
        self.inputs = queue.Queue()
        self.requests_outputs = {}
        self.inference_event = threading.Event()
        self.lock = threading.Lock()

        del configs

    def add_requests(self, requests):
        for request_id, request_input, request_output in requests:
            self.inputs.put(request_input)
            with self.lock:
                self.requests_outputs[request_id] = request_output
        self.inference_event.set()

    def step(self):
        requests_output_pos = []
        merged_tokens = []
        spk_id = None
        pos = 0

        t0 = datetime.datetime.now()
        print(f'thc: flow_and_hift: start: {t0:%H:%M:%S.%f}')

        while True:
            try:
                request_input = self.inputs.get_nowait()
            except queue.Empty:
                break
            request_id = request_input['request_id']
            merged_tokens.extend([self.silent_tokens[0]] * self.split_token_len)
            pos += self.split_token_len
            merged_tokens.extend(request_input['tokens'])
            start_pos = pos + request_input['start_pos']
            end_pos = pos + request_input['end_pos']
            pos += request_input['end_pos']
            pos += 0 if request_input['finalize'] else self.flow.pre_lookahead_len
            # print(f'thc: {request_id}, merged_tokens = {len(merged_tokens)}, start_pos = {start_pos}, end_pos = {end_pos}, pos = {pos}')

            spk_id = spk_id or request_input['spk_id']
            assert request_input['spk_id'] == spk_id

            requests_output_pos.append({ \
                'request_id': request_id,
                'first': request_input['first'],
                'finalize': request_input['finalize'],
                'token_start_pos': start_pos, 'token_end_pos': end_pos,
                'voice_start_pos': int(start_pos * self.voice_sample_rate / self.token_sample_rate),
                'voice_end_pos': int(end_pos * self.voice_sample_rate / self.token_sample_rate),
            })
        assert pos == len(merged_tokens), f'pos = {pos}, len(merged_tokens) = {len(merged_tokens)}'

        tokens = merged_tokens
        tokens = torch.tensor(tokens, dtype=torch.int32, device=self.device).unsqueeze(dim=0)
        token_len = torch.tensor([tokens.shape[1]], dtype=torch.int32, device=self.device)
        flow_prompt_speech_token = self.prompt_wave_info[spk_id]['flow_prompt_speech_token']
        flow_prompt_speech_token_len = self.prompt_wave_info[spk_id]['flow_prompt_speech_token_len']
        prompt_speech_feat = self.prompt_wave_info[spk_id]['prompt_speech_feat']
        prompt_speech_feat_len = self.prompt_wave_info[spk_id]['prompt_speech_feat_len']
        flow_embedding = self.prompt_wave_info[spk_id]['flow_embedding']
        stream = True
        finalize = True
        n_timesteps = 5 if all([v['first'] for v in requests_output_pos]) else 10

        tts_mel, _ = self.flow.inference(token=tokens,
                                            token_len=token_len,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_token_len=flow_prompt_speech_token_len,
                                            prompt_feat=prompt_speech_feat,
                                            prompt_feat_len=prompt_speech_feat_len,
                                            embedding=flow_embedding,
                                            streaming=stream,
                                            finalize=finalize,
                                            n_timesteps=n_timesteps)

        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)

        dt = (datetime.datetime.now() - t0).total_seconds()
        print(f'thc: flow_and_hift: end: {dt:.3f}, tokens: {len(merged_tokens)}, tts_speech: {tts_speech.shape}')

        for n_output in range(len(requests_output_pos)):
            output_pos = requests_output_pos[n_output]
            request_id = output_pos['request_id']
            finalize = output_pos['finalize']
            token_start_pos = output_pos['token_start_pos']
            token_end_pos = output_pos['token_end_pos']
            voice_start_pos = output_pos['voice_start_pos']
            voice_end_pos = output_pos['voice_end_pos']
            part_tts_speech = tts_speech[:, voice_start_pos : voice_end_pos]
            hift_output = {'tts_speech': part_tts_speech, 'finalize': finalize}
            self.requests_outputs[request_id].put(hift_output)

            # print(f'thc: n = {n_output}, {request_id}, finalize = {finalize}, '
            #       f'token_start_pos = {token_start_pos}, token_end_pos = {token_end_pos}, '
            #       f'voice_start_pos = {voice_start_pos}, voice_end_pos = {voice_end_pos}, '
            #       f'part_tts_speech.shape = {part_tts_speech.shape}')

            assert part_tts_speech.shape[1] == (token_end_pos - token_start_pos) * self.voice_sample_rate / self.token_sample_rate


    def inference(self):
        while True:
            self.inference_event.wait()
            self.inference_event.clear()

            if self.inputs.empty():
                continue

            self.step()

class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        # print(f'thc: CosyVoiceServiceImpl, PID = {os.getpid()}, TID = {threading.get_ident()}')

        self.args = args  # 保存args引用
        # 注意：AutoModel(load_vllm=True) 会在此处启动 vLLM 子进程
        # 支持进程池参数
        self.cosyvoice = AutoModel(
            model_dir=args.model_dir,
            load_vllm=True,
            speaker_info_dir=args.speaker_info_dir,
            graph_mode=args.graph_mode
        )
        logging.info(f'grpc service initialized, graph_mode={args.graph_mode}')

        # taohouchao
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.sos_emb = {}
        self.task_id_emb = {}
        self.prompt_text_emb = {}
        self.prompt_speech_token_emb = {}
        self.wait_time = 0.01
        self.cosyvoice.model.llm.sample_rate = self.cosyvoice.sample_rate / 960

        self.requests_info = {}
        self.uuids = [None for _ in range(args.max_conc)]
        self.inference_event = threading.Event()
        self.inference_work_thread = threading.Thread(target=self.inference_work, daemon=True)
        self.inference_work_thread.start()

        self.flow_and_hift = Flow_And_Hift(device=self.device,
                                           fp16=False,
                                           model_dir=args.model_dir,
                                           speaker_info_dir=args.speaker_info_dir,
                                           graph_mode=args.graph_mode)

        # self.flow_and_hift_inference_work_thread = threading.Thread(target=self.flow_and_hift.inference, daemon=True)
        # self.flow_and_hift_inference_work_thread.start()

        # 如果不使用进程池，在主进程构建图模式
        if args.graph_mode:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            # config.mode = "reduce-overhead"
            #config.experimental_config.tiling_schedule_optimize = True
            #config.experimental_config.frozen_parameter = True
            npu_backend = tng.get_npu_backend(compiler_config=config)
            self.flow_and_hift.flow.decoder.estimator.forward = torch.compile(
                self.flow_and_hift.flow.decoder.estimator.forward,
                dynamic=True,
                fullgraph=True,
                backend=npu_backend
            )
            logging.info('compile graph success in main process')

        # 启动vllm profile（通过参数控制）
        if args.enable_vllm_profile:
            assert False
            try:
                self.cosyvoice.model.llm.vllm.start_profile()
                logging.info('vLLM profile started')
            except Exception as e:
                logging.warning(f'Failed to start vLLM profile: {e}') 

    def inference_work(self):
        cosyvoice = self.cosyvoice
        frontend = cosyvoice.frontend
        model = cosyvoice.model
        llm = model.llm
        flow = model.flow
        hift = model.hift

        print(f'thc: inference_work: start')

        while True:
            print(f'thc: inference_work: wait')

            self.inference_event.wait()

            if llm.vllm.get_num_unfinished_requests() == 0:
                continue

            print(f'thc: inference_work: llm.vllm.get_num_unfinished_requests() = {llm.vllm.get_num_unfinished_requests()}')

            n_step = 0

            t0 = datetime.datetime.now()

            while True:
                if llm.vllm.get_num_unfinished_requests() > 0:
                    self.inference_event.clear()
                    requests_output = llm.vllm.step()

                    llm_section_latency_0 = (datetime.datetime.now() - t0).total_seconds()

                    # print(f'n_step = {n_step}, len(requests_output) = {len(requests_output)}, {[len(request_output.outputs[0].token_ids) for request_output in requests_output]}')

                    n_step += 1

                    for request_output in requests_output:
                        token_ids = request_output.outputs[0].token_ids
                        this_uuid = request_output.request_id
                        request_info = self.requests_info[this_uuid]
                        if token_ids[-1] in llm.stop_token_ids:
                            request_info['llm']['output'] = token_ids[: -1]
                        else:
                            request_info['llm']['output'] = token_ids

                        request_info['first'] = len(request_info['llm']['sections_len']) == 0
                        request_info['finalize'] |= (token_ids[-1] in llm.stop_token_ids) or (len(request_info['llm']['output']) >= request_info['llm']['max_len'])
                        request_info['llm']['times']['start'] = request_info['llm']['times']['start'] or datetime.datetime.now()
                        if request_info['finalize']:
                            request_info['llm']['times']['end'] = request_info['llm']['times']['end'] or datetime.datetime.now()

                    first_flag = []
                    non_first_flag = []
                    finalize_flag = []
                    flow_and_hift_requests = []
                    with self.lock:
                        for request_info in self.requests_info.values():
                            if len(request_info['llm']['sections_len']) == len(request_info['hift']['voices_dt']):
                                new_len = len(request_info['llm']['output']) - sum(request_info['llm']['sections_len'])
                                if len(request_info['llm']['sections_len']) == 0:
                                    first_flag.append(new_len >= flow.pre_lookahead_len + 20)
                                elif len(request_info['llm']['sections_len']) == 1:
                                    non_first_flag.append(new_len >= flow.pre_lookahead_len + 25)
                                elif len(request_info['llm']['sections_len']) == 2:
                                    non_first_flag.append(new_len >= flow.pre_lookahead_len + 50)
                                else:
                                    non_first_flag.append(new_len >= flow.pre_lookahead_len + 100)
                            finalize_flag.append(request_info['finalize'])
                        # print(f'thc: inference_work: first_flag = {first_flag}, non_first_flag = {non_first_flag}, finalize_flag = {finalize_flag}')

                        for request_info in self.requests_info.values():
                            request_idx = request_info['idx']
                            this_uuid = request_info['uuid']
                            flow_input = None
                            if (len(first_flag) > 0 and all(first_flag)) or \
                            (any(first_flag) == False and any(non_first_flag)) or \
                            all(finalize_flag):
                                if len(request_info['llm']['output']) - sum(request_info['llm']['sections_len']) >= flow.pre_lookahead_len + 10 or \
                                    (request_info['finalize'] and len(request_info['llm']['output']) - sum(request_info['llm']['sections_len']) > 0):
                                    abs_start_pos = max(0, sum(request_info['llm']['sections_len']) - 10)
                                    abs_end_pos = len(request_info['llm']['output'])
                                    part_start_pos = min(10, sum(request_info['llm']['sections_len']))
                                    pre_lookahead_len = 0 if request_info['finalize'] else flow.pre_lookahead_len
                                    part_len = len(request_info['llm']['output']) - sum(request_info['llm']['sections_len']) - pre_lookahead_len
                                    part_end_pos = part_start_pos + part_len

                                    flow_input = {'request_id': this_uuid,
                                                'spk_id': request_info['spk_id'],
                                                'tokens': request_info['llm']['output'][abs_start_pos : abs_end_pos],
                                                'start_pos': part_start_pos,
                                                'end_pos': part_end_pos, 
                                                'first': request_info['first'],
                                                'finalize': request_info['finalize']}
                                    request_info['flow']['inputs'].put(flow_input)
                                    request_info['flow']['times'].append({'start': datetime.datetime.now()})
                                    request_info['llm']['sections_len'].append(part_len)

                                    # print(f'thc: inference_work: {request_idx}: {this_uuid}, {request_info["flow"]["times"][-1]["start"]:%H:%M:%S.%f}, '
                                    #     f'flow_input: abs_start_pos = {abs_start_pos}, abs_end_pos = {abs_end_pos}, '
                                    #     f'flow_input: start_pos = {flow_input["start_pos"]}, end_pos = {flow_input["end_pos"]}, '
                                    #     f'first = {flow_input["first"]}, finalize = {flow_input["finalize"]}')
                                    flow_and_hift_requests.append([this_uuid, flow_input, request_info['hift']['outputs']])

                    if len(flow_and_hift_requests) > 0:
                        llm_section_latency_1 = (datetime.datetime.now() - t0).total_seconds()
                        print(f'thc: llm_section_latency = {llm_section_latency_0:.3f}, {llm_section_latency_1:.3f}')
                        self.flow_and_hift.add_requests(flow_and_hift_requests)
                        self.flow_and_hift.step()
                        t0 = datetime.datetime.now()
                else:
                    with self.lock:
                        release_flag = [request_info['finalize'] and len(request_info['llm']['sections_len']) == len(request_info['hift']['voices_dt']) for request_info in self.requests_info.values()]
                    # print(f'thc: release_flag: {release_flag}')

                    if all(release_flag):
                        with self.lock:
                            for request_info in self.requests_info.values():
                                # print(f'thc: {request_info["idx"]}: put release')
                                request_info['release'].put(True)
                        break

            gc.collect()
            if self.device.type == 'npu':
                torch_npu.npu.empty_cache()

    def submit_inference(self, tts_text, spk_id, stream_flag):
        cosyvoice = self.cosyvoice
        frontend = cosyvoice.frontend
        model = cosyvoice.model
        llm = model.llm
        flow = model.flow
        hift = model.hift

        this_uuid = str(uuid.uuid1())

        wave_info = cosyvoice.promote_wave_info[spk_id]

        if spk_id not in frontend.spk2info:
            with self.lock:
                if spk_id not in frontend.spk2info:
                    cosyvoice.add_zero_shot_spk(wave_info['prompt_text'], wave_info['prompt_wav'], spk_id)

        # todo iter text
        # tts_text = frontend.text_normalize(tts_text, split=True, text_frontend=frontend.text_frontend)[0]
        tts_texts = frontend.text_normalize(tts_text, split=True, text_frontend=frontend.text_frontend)
        tts_text = ''.join(tts_texts)
        model_input = frontend.frontend_zero_shot(tts_text, wave_info['prompt_text'], wave_info['prompt_wav'], cosyvoice.sample_rate, spk_id)

        #    for model_output in self.model.tts(**model_input, stream=stream, speed=speed):-
        text = model_input['text']
        text_len = model_input['text_len']
        prompt_text = model_input['prompt_text']
        llm_prompt_speech_token = model_input['llm_prompt_speech_token']

        # todo different req different spk_id
        if self.prompt_speech_token_emb.get(spk_id, None) is None:
            with self.lock:
                if self.prompt_speech_token_emb.get(spk_id, None) is None:
                    self.sos_emb[spk_id] = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
                    self.task_id_emb[spk_id] = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
                    self.prompt_text_emb[spk_id] = llm.llm.model.model.embed_tokens(prompt_text)
                    self.prompt_speech_token_emb[spk_id] = llm.speech_embedding(llm_prompt_speech_token)

        sos_emb = self.sos_emb[spk_id]
        task_id_emb = self.task_id_emb[spk_id]
        prompt_text_emb = self.prompt_text_emb[spk_id]
        prompt_speech_token_emb = self.prompt_speech_token_emb[spk_id]
        text_emb = llm.llm.model.model.embed_tokens(text)
        lm_input = torch.concat([sos_emb, prompt_text_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)
        # print(f'thc: submit_inference: lm_input: [{lm_input.dtype}, lm_input.shape: {lm_input.shape}]')

        min_token_text_ratio = 2
        max_token_text_ratio = 20
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # self.llm.inference_wrapper(lm_input, sampling, min_len, max_len, uuid)
        sampling = 25
        sampling_params = vllm.SamplingParams(top_k=sampling,
                                              stop_token_ids=llm.stop_token_ids,
                                              min_tokens=min_len,
                                              max_tokens=max_len)

        now = datetime.datetime.now()
        request_info = {'idx': None, 'uuid': this_uuid, 'submit_time': now,
                        'spk_id': spk_id,
                        'model_input': model_input, 'first': False, 'finalize': False, 'release': queue.Queue(),
                        'llm':  {'output': [], 'max_len': max_len, 'sections_len': [], 'times': {'start': None, 'end': None}},
                        'flow': {'has_first': False, 'inputs': queue.Queue(), 'times': []},
                        'hift': {'outputs': queue.Queue(), 'voices_dt': [], 'times': []}
                       }

        with self.lock:
            for n in range(len(self.uuids)):
                uuid_ = self.uuids[n]
                other_requests_info = self.requests_info.get(uuid_, None)
                if other_requests_info is not None and (now - other_requests_info['submit_time']).total_seconds() > 600:
                    print(f'thc: submit_inference: request_idx: {request_idx}, {this_uuid}, timeout')
                    self.requests_info.pop(uuid_)
                    self.uuids[n] = None

            request_idx = [n for n in range(args.max_conc) if self.uuids[n] is None]
            assert len(request_idx) > 0
            request_idx = request_idx[0]
            self.uuids[request_idx] = this_uuid

            request_info['idx'] = request_idx
            self.requests_info[this_uuid] = request_info
            llm.vllm.add_request(this_uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16)}, sampling_params)
        self.inference_event.set()

        print(f'thc: submit_inference: request_idx: {request_idx}, {this_uuid}, tts_text = {tts_text}, spk_id = {spk_id}')

        return request_info

    def request_inference(self, tts_text, spk_id, stream_flag):
        # print(f'thc: request_inference: , tts_text = "{tts_text}"')
        request_info = self.submit_inference(tts_text, spk_id, stream_flag)

        this_uuid = request_info['uuid']
        request_idx = request_info['idx']
        model_input = request_info['model_input']
        flow_prompt_speech_token = model_input['flow_prompt_speech_token']
        flow_prompt_speech_token_len = model_input['flow_prompt_speech_token_len']
        prompt_speech_feat = model_input['prompt_speech_feat']
        prompt_speech_feat_len = model_input['prompt_speech_feat_len']
        flow_embedding = model_input['flow_embedding']
        stream = True
        speed = 1.0

        cosyvoice = self.cosyvoice
        frontend = cosyvoice.frontend
        model = cosyvoice.model
        llm = model.llm
        flow = model.flow
        hift = model.hift

        while True:
            hift_output = request_info['hift']['outputs'].get(timeout=600)
            tts_speech = (hift_output['tts_speech'].view(-1).cpu().numpy() * (2 ** 15)).astype(np.int16)
            voice_dt = tts_speech.shape[0] / cosyvoice.sample_rate
            request_info['hift']['voices_dt'].append(voice_dt)
            request_info['hift']['times'].append({'end': datetime.datetime.now()})

            # print(f'thc: {this_uuid}, {tts_speech.shape}, {voice_dt}')

            yield tts_speech

            if hift_output['finalize']:
                # print(f'thc: request_inference: {request_idx}: submit: {request_info["submit_time"]:%H:%M:%S.%f}, '
                #       f'llm: sections_len: {request_info["llm"]["sections_len"]}, voices_dt: {request_info["hift"]["voices_dt"]}, '
                #       f'calc_dt: {(request_info["llm"]["times"]["end"] - request_info["llm"]["times"]["start"]).total_seconds():.3f}')

                for n_section in range(len(request_info["llm"]["sections_len"])):
                    t2w_calc_dt = (request_info['hift']['times'][n_section]['end'] - request_info['flow']['times'][n_section]['start']).total_seconds()
                    voice_dt = request_info["hift"]["voices_dt"][n_section]
                    if n_section == 0:
                        latency = (request_info['hift']['times'][n_section]['end'] - request_info["submit_time"]).total_seconds()
                        prev_voice_dt = -1
                    else:
                        latency = (request_info['hift']['times'][n_section]['end'] - request_info['hift']['times'][n_section - 1]['end']).total_seconds()
                        prev_voice_dt = request_info["hift"]["voices_dt"][n_section - 1]
                    rtf = latency / voice_dt
                    # print(f'thc: request_inference: {request_idx}: section = {n_section}: t2w_calc_dt = {t2w_calc_dt:.3f}, voice_dt = {voice_dt:.3f}, '
                    #       f'prev_voice_dt = {prev_voice_dt:.3f}, latency = {latency:.3f}, rtf = {rtf:.3f}')

                release = request_info['release'].get(timeout=600)
                with self.lock:
                    self.requests_info.pop(this_uuid)
                    self.uuids[request_idx] = None
                    latency = (datetime.datetime.now() - request_info["submit_time"]).total_seconds()
                    first_latency = (request_info['hift']['times'][0]['end'] - request_info["submit_time"]).total_seconds()
                    print(f'thc: request_inference: pop: {request_idx}: {this_uuid}, submit: {request_info["submit_time"]:%H:%M:%S.%f}, first_latency: {first_latency:.3f}, latency: {latency:.3f}')
                return
                

    def Inference(self, request, context):
        # print(f'thc: Inference: context = {context}, PID = {os.getpid()}, TID = {threading.get_ident()}')
        # print(f'thc: Inference: tts_text = {request.zero_shot_by_id_request.tts_text}, spk_id = {request.zero_shot_by_id_request.spk_id}')

        if 1:
            tts_text = request.zero_shot_by_id_request.tts_text
            spk_id = request.zero_shot_by_id_request.spk_id
            stream_flag = request.zero_shot_by_id_request.stream

            for tts_speech in self.request_inference(tts_text, spk_id, stream_flag):
                response = cosyvoice_pb2.Response()
                response.tts_audio = tts_speech.tobytes()
                yield response
        else:
            if request.HasField('zero_shot_by_id_request'):
                logging.info('get zero_shot_by_id inference request')
                model_output = self.cosyvoice.inference_zero_shot_by_id(
                    request.zero_shot_by_id_request.tts_text,
                    request.zero_shot_by_id_request.spk_id,
                    stream=request.zero_shot_by_id_request.stream
                )
            else:
                assert False
                logging.info('get instruct inference request')
                model_output = self.cosyvoice.inference_instruct(request.instruct_request.tts_text,
                                                                request.instruct_request.spk_id,
                                                                request.instruct_request.instruct_text)

            logging.info('send inference response')

            count=0
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                print(f'thc: Inference: tts_speech = {i["tts_speech"].shape}')
                yield response
                count+=1
                if count==4 and self.args.enable_vllm_profile:
                    assert False
                    try:
                        self.cosyvoice.model.llm.vllm.stop_profile()
                        logging.info('vLLM profile stopped')
                    except Exception as e:
                        logging.warning(f'Failed to stop vLLM profile: {e}')


def main():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--max_conc',
                        type=int,
                        default=4)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--graph_mode', 
                        action='store_true', 
                        help='whether to use graph mode')
    parser.add_argument('--speaker_info_dir',
                        type=str,
                        default='./',
                        help='directory containing speaker info files (default: model_dir/speaker_info)')
    parser.add_argument('--enable_vllm_profile',
                        action='store_true',
                        help='whether to enable vLLM profiling (default: False)')
    args = parser.parse_args()
    logging.info(f"args：{args}")
    main()
