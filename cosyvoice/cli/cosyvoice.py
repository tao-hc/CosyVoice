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
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.utils.file_utils import load_wav
import datetime
import torchaudio
import uuid


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) == CosyVoiceModel, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        _has_accelerator = torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())
        if not _has_accelerator and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda/npu device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_wav, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert self.__class__.__name__ == 'CosyVoice', 'inference_instruct is only implemented for CosyVoice!'
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_wav, prompt_wav, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_wav, prompt_wav, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        _has_accelerator = torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())
        if not _has_accelerator and (load_jit is True or load_trt is True or load_vllm is True or fp16 is True):
            load_jit, load_trt, load_vllm, fp16 = False, False, False, False
            logging.warning('no cuda/npu device, set load_jit/load_trt/load_vllm/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()


class CosyVoice3(CosyVoice2):

    def __init__(self, model_dir, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1,
                 speaker_info_dir=None, graph_mode=False):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice3Model, 'do not use {} for CosyVoice3 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v3.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        _has_accelerator = torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())
        if not _has_accelerator and (load_trt is True or fp16 is True):
            load_trt, fp16 = False, False
            logging.warning('no cuda/npu device, set load_trt/fp16 to False')
        self.model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'], fp16,
                                     model_dir=model_dir, graph_mode=graph_mode)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning('DiT tensorRT fp16 engine have some performance issue, use at caution!')
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

        # 设置speaker_info目录
        self.speaker_info_dir = speaker_info_dir or os.path.join(model_dir, 'speaker_info')
        if not os.path.exists(self.speaker_info_dir):
            assert False
            logging.warning(f'speaker_info_dir {self.speaker_info_dir} does not exist')

        # 预加载所有说话人信息
        self.promote_wave_info = {}
        self._preload_all_wave_info()

    def _preload_all_wave_info(self):
        """预加载所有说话人信息（从统一的JSON文件）"""
        if not os.path.exists(self.speaker_info_dir):
            return

        import json

        # 查找speaker_info.json文件
        speaker_info_file = os.path.join(self.speaker_info_dir, 'speaker_info.json')
        if not os.path.exists(speaker_info_file):
            logging.warning(f'speaker_info.json not found in {self.speaker_info_dir}')
            return

        try:
            with open(speaker_info_file, 'r', encoding='utf-8') as f:
                all_speakers = json.load(f)

            # 加载每个说话人的音频
            for spk_id, wave_info in all_speakers.items():
                try:
                    # 加载音频文件
                    prompt_wav_path = os.path.join(self.speaker_info_dir, wave_info['prompt_wav'])
                    if os.path.exists(prompt_wav_path):
                        prompt_wav = load_wav(wave_info['prompt_wav'], 16000)
                        wave_info['prompt_wav'] = prompt_wav
                        self.promote_wave_info[spk_id] = wave_info
                        logging.info(f"Loaded speaker {spk_id} from {prompt_wav_path}, promote={wave_info['prompt_text']}")
                    else:
                        logging.error(f'Audio file not found for speaker {spk_id}: {prompt_wav_path}')
                except Exception as e:
                    logging.error(f'Failed to load speaker {spk_id}: {e}')

            logging.info(f'Preloaded {len(self.promote_wave_info)} speakers from {speaker_info_file}')

        except Exception as e:
            logging.error(f'Failed to load {speaker_info_file}: {e}')

    def inference_zero_shot_by_id(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        """使用预定义的说话人ID执行zero_shot推理"""
        # 从缓存中获取说话人信息
        if spk_id not in self.promote_wave_info:
            raise ValueError(f'Speaker ID {spk_id} not found. Available IDs: {list(self.promote_wave_info.keys())}')
        wave_info = self.promote_wave_info[spk_id]
        
        # 0.save audio
        #audio_uuid = str(uuid.uuid1())
        # 缓存相关信息
        if spk_id not in self.frontend.spk2info:
            self.add_zero_shot_spk(wave_info['prompt_text'], wave_info['prompt_wav'], spk_id)

        # 使用zero_shot接口进行推理
        # 1.save audio
        #res = torch.tensor([])
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            print(f'thc: inference_zero_shot_by_id: {i}')
            model_input = self.frontend.frontend_zero_shot(i, wave_info['prompt_text'], wave_info['prompt_wav'], self.sample_rate, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                # 2.save audio
                #res = torch.cat((res, model_output['tts_speech']), dim=1)
                yield model_output
                start_time = time.time()
        # 3.save audio
        #now_time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        #torchaudio.save(f'./server-audio/{now_time}-{audio_uuid}.wav', res, self.sample_rate)

    def reload_wave_info(self):
        """重新加载所有说话人信息"""
        self.promote_wave_info.clear()
        self._preload_all_wave_info()

    def get_available_spk_ids(self):
        """获取所有可用的说话人ID"""
        return list(self.promote_wave_info.keys())


def AutoModel(**kwargs):
    if not os.path.exists(kwargs['model_dir']):
        kwargs['model_dir'] = snapshot_download(kwargs['model_dir'])
    if os.path.exists('{}/cosyvoice.yaml'.format(kwargs['model_dir'])):
        return CosyVoice(**kwargs)
    elif os.path.exists('{}/cosyvoice2.yaml'.format(kwargs['model_dir'])):
        return CosyVoice2(**kwargs)
    elif os.path.exists('{}/cosyvoice3.yaml'.format(kwargs['model_dir'])):
        return CosyVoice3(**kwargs)
    else:
        raise TypeError('No valid model type found!')
