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
- 无需密钥的 Open-Meteo 当前天气/小时预报与 NOAA METAR/TAF 航空气象；美国机场使用 FAA NAS Status 当前运行事件，其他机场可使用 AirLabs 航班样本或 ADSB.lol 飞机密度代理；新闻来自无需密钥的 GDELT。
- 天气和新闻卡片可进入独立的中英双语详情页；详情页支持手动刷新，并分别每 10 分钟、15 分钟自动刷新。
- 外部服务超时、无数据或额度不足时，自动使用明确标注的历史/模型平均值或中性新闻值。
- FastAPI、OpenAPI 文档、CLI 训练入口、时间切分评估和自动测试。

The UI always labels external context as `live`, `forecast`, `proxy`, `historical`, or `neutral`. Missing policy information remains `unknown`; the service never converts “unknown” into “not included.”

## 免费数据模式 / Strict-free data mode

| 数据 | 默认来源 | 无法获取时 |
| --- | --- | --- |
| 天气 | 2 小时内使用 Open-Meteo 当前模型天气并结合 NOAA METAR/TAF；2–30 小时结合小时预报与 TAF；其后至 16 天使用 Open-Meteo 小时预报 | 明确标记的合成训练集同月平均值 |
| 机场运行 | 美国机场优先使用无需密钥的 FAA NAS Status 当前事件；其他机场配置 AirLabs 免费密钥后使用计划出港样本，否则显示 ADSB.lol 当前飞机密度代理 | 与目标起飞时刻不匹配时，模型使用明确标记的训练平均值/合成先验；当前快照仍单独展示 |
| 时事新闻 | 无需密钥的 GDELT DOC 2.0 最近 7 天中断新闻，按最新观察时间排序；DOC 不可用时使用官方 GAL 滚动 RSS | 先使用最多 6 小时的带标签缓存并降低影响；没有缓存时返回中性值且不生成新闻 |
| 航线航司 | 配置密钥时使用 AirLabs routes | 60 家全球模型比较目录 |

AirLabs 密钥是可选项。没有任何密钥时项目仍可运行。Open-Meteo 公共免费端点限非商业用途并采用 CC BY 4.0；GDELT 允许免费使用，但使用或再分发其数据时须注明并链接 GDELT。免费额度、覆盖和条款可能变化，公开部署前应重新检查 [Open-Meteo 官方条款](https://open-meteo.com/en/terms)、[GDELT 项目说明](https://www.gdeltproject.org/about.html)与其他来源要求，详见 [运行时数据与回退](docs/runtime-context.md)。

机场运行采用两层语义：`context.operations` 顶层是实际送入目标起飞时刻预测的信号，`context.operations.current_snapshot` 是查询时刻的机场快照。两者可能不同。例如，数周后的航班会使用训练平均值/合成先验进行预测，同时仍可展示今天的 FAA 事件或飞机密度。FAA 的 `freeForm` 限制（例如只影响部分通航飞机）只作为低权重当前提示，不能被解释为整个商业机场关闭；ADSB.lol 也只表示机场附近飞机密度，不提供真实延误、取消或地面停飞数据。

Airport operations use two distinct layers: the top-level `context.operations` is the target-departure signal actually used by the model, while `context.operations.current_snapshot` describes conditions at request time. FAA scoped restrictions are not treated as full commercial-airport closures, and ADSB.lol aircraft density is only a traffic proxy—not an actual delay, cancellation, or ground-stop feed.

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

先在主页完成一次比较，再点击天气或新闻卡片的“查看详情 / View details”。详情链接在当前标签页打开，URL 包含航线、出发时间和语言，因而可以复制分享；返回主页时，浏览器会话中的表单、排序和最近一次比较结果会恢复。

macOS 或 Linux 激活环境：

```bash
source .venv/bin/activate
```

### 可选免费 AirLabs 密钥 / Optional free AirLabs key

免费 AirLabs 密钥用于非美国机场的计划出港样本和航线航司确认；没有密钥时，美国机场仍可使用 FAA，其他机场则显示 ADSB.lol 代理并在需要时使用模型先验。密钥只保存在本地环境变量，不要提交到 GitHub：

```powershell
$env:AIRLABS_API_KEY="your-free-key"
python -m flight_forecaster serve --model-dir artifacts/demo
```

完全离线模式：

```powershell
$env:EXTERNAL_CONTEXT_ENABLED="0"
python -m flight_forecaster serve --model-dir artifacts/demo
```

可参考 [.env.example](.env.example)。当前启动命令直接读取进程环境变量，不会自动加载 `.env`；请在启动服务的同一个终端设置变量，并且不要提交密钥文件。

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
- 天气、机场运行和新闻信号及来源、状态、时间；机场运行还区分目标时刻信号与 `current_snapshot` 当前快照；
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

## 详情页面与接口 / Detail pages and APIs

详情页面由主页生成带查询参数的可分享链接，并在当前标签页打开：

- `GET /details/weather`：同时展示出发机场计划起飞与到达机场预计抵达时刻的当前天气、目标时刻预报、前后 12 小时趋势、风险拆解，以及 NOAA METAR/TAF 原始报文、独立可用性元数据和保守的中英双语解释；页面可手动刷新并每 10 分钟自动刷新。
- `GET /details/news`：展示最近 7 天最多 20 篇匹配报道的原始标题、来源、语言、GDELT 索引时间、风险类别、命中词和时效权重，并单独标明详情页文章风险与实际进入模型的 `model_signal`；页面可手动刷新并每 15 分钟自动刷新。标题不做机器翻译。

两页调用相同的三字段请求：

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "departure_time": "2026-09-15T08:00:00-04:00"
}
```

- `POST /v1/context/weather-detail`
- `POST /v1/context/news-detail`

The detail pages use the same `中文 / English` switch as the dashboard. Weather refreshes every 10 minutes and news every 15 minutes; both also provide a manual refresh button. A detail-page refresh fetches the API again, although the server may legitimately return a short-lived provider cache.

## 新闻如何进入预测 / How news affects predictions

系统通过无需 API 密钥的 GDELT DOC 2.0 查询最近 7 天内与起点、终点及航空中断词相关的文章，并使用 `DateDesc` 按 GDELT 最新观察时间排序。DOC 请求失败时会尝试 GDELT 官方 GAL RSS：该 RSS 每分钟更新，保留最近约 15 分钟的文章。相同航线的成功结果缓存 15 分钟；实时来源暂时失败时可返回不超过 6 小时、明确标为 `historical` 且降低模型影响的旧缓存；没有可用缓存才返回值 0、状态 `neutral`，且绝不虚构标题。

仅对真实返回的标题进行保守评分，所得 `news_disruption_index` 是票价和准点模型的输入特征。主页卡片最多显示 5 篇去重后的来源文章，新闻详情页最多显示 20 篇。页面中的文章时间是 GDELT 观察 / 索引该文章的时间，并不保证等于媒体标注的准确发布时间；标题保留来源语言，系统不伪装成机器翻译标题。

The service queries the no-key GDELT DOC 2.0 API for route-related disruption coverage from the previous seven days and requests `DateDesc` ordering by GDELT's latest observed time. If DOC fails, it tries GDELT's official GAL RSS feed, which updates every minute and rolls over roughly the latest 15 minutes. A successful route result is fresh-cached for 15 minutes. When live providers fail, a cache no older than six hours may be returned as `historical` with reduced model influence; without a usable cache the signal is `neutral`, zero, and contains no invented headlines.

Only real returned titles are scored, and the bounded `news_disruption_index` feeds both models. The dashboard card shows up to five deduplicated articles; the detail page shows up to 20. Article times are GDELT observed/indexed times and are not guaranteed to be exact publisher timestamps; headlines remain in their source language. The news relationship in the demo training data is synthetic. Article relevance does not prove that a particular flight will be affected, and the feature should be retrained and calibrated with lawful, representative historical snapshots before production use.

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
- FAA NAS Status 只覆盖美国且只表示其列出的当前事件；“没有列出事件”不保证机场所有航班完全正常。AirLabs 免费结果可能受时间窗口、配额和最多 50 行样本限制。
- ADSB.lol 只提供飞机位置/密度代理，不能用于声称真实延误率、取消率、机场容量或官方地面管制状态。
- NOAA、Open-Meteo、AirLabs、ADSB.lol、GDELT 和 OurAirports 的覆盖、频率、许可与可用性可能变化。
- 不应将输出用于自动购票、拒绝退款、差别定价或无人复核的重大财务/安全决策。

## License

代码以 [MIT License](LICENSE) 发布。第三方数据不随 MIT 许可证重新授权；使用者必须遵守各来源的当前条款与署名要求。
