---
comments: true
---

# 多语种语音识别产线使用教程

## 1. 多语种语音识别产线介绍
语音识别是一种先进的工具，它能够将人类口述的多种语言自动转换为相应的文本或命令。该技术还在智能客服、语音助手、会议记录等多个领域发挥着重要作用。多语种语音识别则可以支持自动进行语种检索，支持多种不同语言的语音的识别。

<p><b>多语种语音识别模型（可选）：</b></p>
<table>
   <tr>
     <th >模型</th>
     <th >模型下载链接</th>
     <th >训练数据</th>
     <th >模型大小</th>
     <th >词错率</th>
     <th >介绍</th>
   </tr>
   <tr>
     <td>whisper_large</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar">whisper_large</a></td>
     <td >680kh</td>
     <td>5.8G</td>
     <td>2.7 (Librispeech)</td>
     <td rowspan="5">Whisper 是 OpenAI 开发的多语言自动语音识别模型，具备高精度和鲁棒性。它采用端到端架构，能处理嘈杂环境音频，适用于语音助理、实时字幕等多种应用。</td>
   </tr>
   <tr>
     <td>whisper_medium</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar">whisper_medium</a></td>
     <td>680kh</td>
     <td>2.9G</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_small</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar">whisper_small</a></td>
     <td>680kh</td>
     <td>923M</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_base</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar">whisper_base</a></td>
     <td>680kh</td>
     <td>277M</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_small</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar">whisper_tiny</a></td>
     <td>680kh</td>
     <td>145M</td>
     <td>-</td>
   </tr>
 </table>

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验多语种语音识别产线的效果。

在本地使用多语种语音识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了 PaddleX 的 wheel 包安装。

### 2.1 本地体验

#### 2.1.1 命令行方式体验
一行命令即可快速体验文档图像预处理产线效果，使用 [示例语音](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline multilingual_speech_recognition \
        --input zh.wav \
        --save_path ./output \
        --device gpu:0
```

相关的参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

```plaintext
{'input_path': 'zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}
```

运行结果参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的结果解释。

#### 2.1.2 Python脚本方式集成

上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multilingual_speech_recognition")
output = pipeline.predict(input="zh.wav")

for res in output:
    res.print()
    res.save_to_json(save_path="./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化 multilingual_speech_recognition 产线对象：具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
</tbody>
</table>

（2）调用 multilingual_speech_recognition 产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据</td>
<td><code>str</code></td>
<td>
<ul>
  <li><b>文件路径</b>，如语音文件的本地路径：<code>/root/data/audio.wav</code></li>
  <li><b>URL链接</b>，如语音文件的网络URL：<a href = "https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav">示例</a></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
  <li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
  <li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
  <li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
  <li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
  <li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: 输入音频存放路径
    - `result`: 识别结果
        -  `text`: 语音识别结果文本
        -  `segments`: 带时间戳的结果文本
            * `id`: ID
            * `seek`: 语音片段指针
            * `start`: 片段开始时间
            * `end`: 片段结束时间
            * `text`: 片段识别文本
            * `tokens`: 片段文本的 token id
            * `temperature`: 变速比例
            * `avg_logprob`: 平均 log 概率
            * `compression_ratio`: 压缩比
            * `no_speech_prob`: 非语音概率
        - `language`: 识别语种

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_audio_basename}.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
</table>


- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。

此外，您可以获取 multilingual_speech_recognition 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config multilingual_speech_recognition --save_path ./my_path
```

若您获取了配置文件，即可对 multilingual_speech_recognition 产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

例如，若您的配置文件保存在 `./my_path/multilingual_speech_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multilingual_speech_recognition.yaml")
output = pipeline.predict(input="zh.wav")
for res in output:
    res.print()
    res.save_to_json("./output/")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改 multilingual_speech_recognition 产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。
