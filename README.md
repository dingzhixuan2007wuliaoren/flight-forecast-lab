# Flight Forecast Lab / 全球航班预测实验室

Flight Forecast Lab 是一个可复现的双任务机器学习项目，用于比较未来行程的**模型票价**与**准点概率**。页面只要求输入出发机场、到达机场和计划起飞时间；距离、飞行时长、天气、机场运行压力和近期新闻均由系统自动解析。

Flight Forecast Lab is a reproducible two-model project for comparing **estimated fares** and **on-time probabilities**. The dashboard only asks for origin, destination, and planned departure; distance, duration, weather, airport operations, and recent news are resolved automatically.

> 重要：默认模型使用确定性的合成演示数据训练。输出不是实时可购买报价，也不是经过全球真实数据验证的生产预测。
>
> Important: the default models are trained on deterministic synthetic demo data. Outputs are neither live bookable fares nor globally validated production forecasts.

## 功能 / Features

- 页面右上角 `中文 / English` 切换，并使用浏览器本地存储记住语言。
- 页面输入的时间自动按出发机场当地 IANA 时区解释，不受用户电脑所在时区影响。
- 119 个内置全球主要机场；其他有效 IATA 机场可通过免费的 OurAirports 目录回退解析。
- 60 家全球主要航司及其可比较舱位场景；配置 AirLabs 免费密钥后，优先使用其返回的航线航司。
- 三类完整排序：直飞优先、低价优先、学生友好优先。
- 学生友好排序严格采用：最低价格 → 已确认免费托运行李 → 已确认实际学生折扣 → 已确认免费改签/退票 → 年龄与验证要求。
- 无需密钥的 Open-Meteo 当前天气/小时预报与 NOAA METAR/TAF 航空气象；另结合 AirLabs 或 ADSB.lol 机场运行信号、GDELT 近期新闻信号。
- 外部服务超时、无数据或额度不足时，自动使用明确标注的历史/模型平均值或中性新闻值。
- FastAPI、OpenAPI 文档、CLI 训练入口、时间切分评估和自动测试。

The UI always labels external context as `live`, `forecast`, `proxy`, `historical`, or `neutral`. Missing policy information remains `unknown`; the service never converts “unknown” into “not included.”

## 免费数据模式 / Strict-free data mode

| 数据 | 默认来源 | 无法获取时 |
| --- | --- | --- |
| 天气 | 2 小时内使用 Open-Meteo 当前模型天气并结合 NOAA METAR/TAF；2–30 小时结合小时预报与 TAF；其后至 16 天使用 Open-Meteo 小时预报 | 明确标记的合成训练集同月平均值 |
| 机场运行 | 出发前 6 小时内使用 AirLabs 或 ADSB.lol 当前信号 | 明确标记的合成训练集机场平均值 |
| 时事新闻 | GDELT DOC 2.0 最近 7 天相关中断新闻 | 中性值，不生成新闻 |
| 航线航司 | 配置密钥时使用 AirLabs routes | 60 家全球模型比较目录 |

AirLabs 密钥是可选项。没有任何密钥时项目仍可运行。Open-Meteo 公共免费端点限非商业用途并采用 CC BY 4.0；免费额度、覆盖和条款可能变化，公开部署前应重新检查[官方条款](https://open-meteo.com/en/terms)与其他来源要求，详见 [运行时数据与回退](docs/runtime-context.md)。

Open-Meteo 的“当前天气”来自约 15 分钟分辨率的天气模型，并非机场传感器实测；NOAA METAR 才是机场观测。页面会分别标记 `open_meteo_current_model`、`noaa_metar`、`forecast`、`live` 或 `proxy`，不会把远期历史平均值冒充实时天气。

Open-Meteo current conditions are model-derived at roughly 15-minute resolution, while NOAA METAR provides airport observations. Departures beyond the free forecast horizon keep using the clearly labelled training-average fallback rather than today's weather.

## 快速运行 / Quick start

支持 Python 3.11–3.13。Windows PowerShell：

```powershell
git clone https://github.com/dingzhixuan2007wuliaoren/flight-forecast-lab.git
cd flight-forecast-lab

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

训练并启动：

```powershell
python -m flight_forecaster train-demo --output artifacts/demo
python -m flight_forecaster serve --model-dir artifacts/demo
```

打开：

- 双语页面 / bilingual dashboard: <http://127.0.0.1:8000/>
- API 文档 / API docs: <http://127.0.0.1:8000/docs>
- 健康检查 / health: <http://127.0.0.1:8000/health>

macOS 或 Linux 激活环境：

```bash
source .venv/bin/activate
```

### 可选免费 AirLabs 密钥 / Optional free AirLabs key

密钥只保存在本地环境变量，不要提交到 GitHub：

```powershell
$env:AIRLABS_API_KEY="your-free-key"
python -m flight_forecaster serve --model-dir artifacts/demo
```

完全离线模式：

```powershell
$env:EXTERNAL_CONTEXT_ENABLED="0"
python -m flight_forecaster serve --model-dir artifacts/demo
```

可参考 [.env.example](.env.example)，但项目不会自动上传或提交 `.env`。

## 比较接口 / Comparison API

`POST /v1/compare`

```json
{
  "origin": "YYZ",
  "destination": "LHR",
  "departure_time": "2026-09-15T08:00:00-04:00"
}
```

响应包括：

- 自动推算的距离和飞行时长；
- 天气、机场运行和新闻信号及来源、状态、时间；
- 每个航司/舱位的模型价格、80% 区间、准点率和风险；
- 行李、学生计划、退改、年龄及验证的保守状态；
- `direct_first`、`lowest_price`、`student_first` 三组完整 ID 排序；
- 中英双语限制说明。

The endpoint always returns the complete global comparison catalog. AirLabs-confirmed direct carriers are marked `provider_confirmed` and ranked first; all other airlines are explicit one-stop `model_scenario` entries. Every cabin is separately labelled `catalog_scenario`, because the free route source does not confirm cabin inventory.

For one-stop model scenarios, the offer duration adds a 90-minute connection and the itinerary on-time probability uses an explicit two-independent-leg assumption (`p²`). The API labels this as `two_leg_independence_scenario`; it is not a confirmed connection itinerary.

网页的无偏移时间按出发机场当地时区解释。直接调用 API 时，无偏移时间同样采用该语义；带偏移的 ISO 8601 时间则按绝对时刻处理。响应中的 `departure_timezone` 可用于核对。

## 单项接口 / Individual endpoints

模型票价：`POST /v1/predict/price`

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "airline": "DL",
  "cabin": "economy",
  "stops": 0,
  "departure_time": "2026-09-15T08:00:00-04:00"
}
```

准点概率：`POST /v1/predict/on-time`

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "airline": "DL",
  "scheduled_departure": "2026-09-15T08:00:00-04:00"
}
```

天气严重度、出发机场拥堵、距离和飞行时长均不再由调用方提交。

## 新闻如何进入预测 / How news affects predictions

系统查询 GDELT 最近 7 天内与起点、终点及航空中断词相关的文章，仅对真实返回的标题进行保守评分。所得 `news_disruption_index` 是票价和准点模型的正式输入特征。页面会显示最多 5 条来源文章；查询失败时值为 0、状态为 `neutral`，不会生成标题。

The news relationship in the demo training data is synthetic. Article relevance does not prove that a particular flight will be affected, and the feature should be retrained and calibrated with lawful, representative historical snapshots before production use.

## 训练与验证 / Training and validation

```powershell
python -m pytest
python -m ruff check src tests
```

使用符合契约的真实 CSV：

```powershell
python -m flight_forecaster train-csv `
  --price-csv data/processed/price.csv `
  --ontime-csv data/processed/on_time.csv `
  --output artifacts/custom
```

详细字段见 [数据契约](docs/data-contracts.md)，真实训练来源见 [数据来源](docs/data-sources.md)，评估边界见 [模型卡](MODEL_CARD.md)。

## 关键限制 / Key limitations

- 合成数据只证明训练、保存、加载和服务链路可工作，不代表真实市场表现。
- 免费模式没有全球实时可售票价；所有页面价格均为模型估价。
- 公共学生计划页面不能证明当前行程已经应用学生专属价；因此目录只标记 `program_available`，不标记“实际折扣已确认”。
- 行李和退改通常依赖具体 fare brand。没有报价级证据时保持 `unknown`。
- NOAA、Open-Meteo、AirLabs、ADSB.lol、GDELT 和 OurAirports 的覆盖、频率、许可与可用性可能变化。
- 不应将输出用于自动购票、拒绝退款、差别定价或无人复核的重大财务/安全决策。

## License

代码以 [MIT License](LICENSE) 发布。第三方数据不随 MIT 许可证重新授权；使用者必须遵守各来源的当前条款与署名要求。
