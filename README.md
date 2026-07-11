# Flight Forecast Lab

Flight Forecast Lab 是一个可复现的双任务机器学习项目：

- **机票价格回归**：根据航线、航司、舱位、经停、行程时长、距离、询价时间和起飞时间，给出未来行程的美元价格点估计与 **80% 预测区间**。
- **准点率分类**：估计航班“准点”的概率。本项目将准点严格定义为：**航班未取消，且到达延误少于 15 分钟**。

项目包含命令行训练入口、FastAPI 推理服务、模型与数据说明，以及一个完全离线的合成数据演示。合成演示用于验证工程流程，**不是经过真实市场数据验证的生产预测器**。

## 当前能力

| 能力 | 输出 | 状态 |
| --- | --- | --- |
| 票价预测 | 点估计 + 80% 预测区间 | 可用；默认演示模型基于合成数据 |
| 准点预测 | 准点概率 + 扰动概率 + 风险级别 | 可用；默认演示模型基于合成数据 |
| 模型训练 | 两个任务一次训练并写入同一模型目录 | 可用 |
| 在线推理 | FastAPI、自动 OpenAPI 文档 | 可用 |
| 真实数据接入 | BTS DB1B/DB1C、BTS On-Time、NOAA 的字段映射与数据契约 | 已定义适配路径；数据需由使用者下载、校验并重新训练 |

## 快速开始

项目支持 Python **3.11–3.13**，建议使用 Python 3.12 并在虚拟环境中安装。当前依赖声明不接受 Python 3.14。项目不需要 API 密钥即可运行合成数据演示。

```powershell
git clone https://github.com/dingzhixuan2007wuliaoren/flight-forecast-lab.git
cd flight-forecast-lab

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS 或 Linux 激活虚拟环境时改用：

```bash
source .venv/bin/activate
```

训练内置演示模型：

```bash
python -m flight_forecaster train-demo --output artifacts/demo
```

启动服务：

```bash
python -m flight_forecaster serve --model-dir artifacts/demo
```

服务启动后可访问：

- 浏览器演示页：<http://127.0.0.1:8000/>
- 交互式 API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>
- 模型信息：<http://127.0.0.1:8000/v1/model-info>

### API 请求

票价预测使用 `POST /v1/predict/price`：

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/price \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "JFK",
    "destination": "LAX",
    "airline": "DL",
    "cabin": "economy",
    "stops": 0,
    "duration_minutes": 365,
    "distance_km": 3983,
    "quote_time": "2026-07-11T12:00:00Z",
    "departure_time": "2026-08-15T13:30:00Z"
  }'
```

响应包含：

```json
{
  "estimated_price_usd": 246.12,
  "interval_80_low_usd": 171.43,
  "interval_80_high_usd": 320.81,
  "days_until_departure": 35.1,
  "model_version": "0.1.0",
  "warning": "条件估价并非实时可购买报价，也不保证最低价。"
}
```

以上数值仅展示响应形状，不是固定预测结果。

准点预测使用 `POST /v1/predict/on-time`：

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/on-time \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "JFK",
    "destination": "LAX",
    "airline": "DL",
    "distance_km": 3983,
    "scheduled_departure": "2026-08-15T13:30:00Z",
    "weather_severity_forecast": 0.25,
    "origin_congestion_index": 0.40
  }'
```

响应包含 `on_time_probability`、其补数 `disruption_probability`、`risk_level`（`low` / `medium` / `high`）、准点定义和模型版本。风险级别不是新的训练标签：服务按准点概率将其分为低、中、高扰动风险。

字段约束和单位见 [数据契约](docs/data-contracts.md)。运行中的 `/docs` 是请求与响应模式的最终依据。

## 验证

```bash
python -m pytest
```

若已经把真实数据转换成契约兼容的 CSV，可用：

```bash
python -m flight_forecaster train-csv \
  --price-csv data/processed/price.csv \
  --ontime-csv data/processed/on_time.csv \
  --output artifacts/custom
```

该命令只负责读取统一契约并训练，不会自动下载 BTS 或 NOAA。模型目录将包含 `model_bundle.joblib`、`metrics.json`、`metadata.json` 和 `report.md`。也可以用 `predict-price` / `predict-on-time` 子命令读取单个 JSON 请求；运行 `python -m flight_forecaster --help` 查看参数。

建议在提交真实模型前同时检查：测试是否通过、时间切分是否正确、模型目录中的训练摘要是否与本次数据一致，以及 API 是否能加载该目录。

## 项目结构

```text
src/flight_forecaster/   Python 包、训练流程、CLI、网页与 API
tests/                   自动化测试
docs/
  data-sources.md        官方数据来源与接入注意事项
  data-contracts.md      训练数据和 API 字段契约
artifacts/               本地训练产物（通常不提交大型二进制文件）
MODEL_CARD.md            模型用途、限制与评估要求
```

## 从演示迁移到真实数据

官方来源：

- [BTS Origin and Destination Survey（DB1B / DB1C）](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data)
- [BTS Reporting Carrier On-Time Performance](https://transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=&gnoyr_VQ=FGJ)
- [NOAA/NCEI GHCN Hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)
- [NOAA Aviation Weather Center Data API](https://aviationweather.gov/data/api/)

真实训练建议按以下顺序进行：

1. 从 BTS 下载 O&D 票价数据。历史时期使用 DB1B；2025 年 7 月起的新制度数据使用 DB1C。
2. 从 BTS Reporting Carrier On-Time Performance 下载航班级运行记录，按本项目定义生成 `on_time` 标签。
3. 按机场和时间拼接 NOAA 天气。训练时只能使用预测时刻已经可知的信息；若使用实际观测天气，必须明确它只适合回溯分析，不能冒充未来天气预报。
4. 先落到 [统一数据契约](docs/data-contracts.md)，完成重复值、单位、时区、缺失值和异常票价检查。
5. 使用按时间向前滚动的训练/验证/测试切分重新训练，并在未见过的较新时间段上报告指标。

官方入口与数据粒度详见 [真实数据来源](docs/data-sources.md)。

## 重要限制

- **合成数据不代表真实世界表现。** 它只能证明训练、保存、加载和服务链路可以工作。
- **公开票价调查不等于实时搜索报价。** DB1B/DB1C 是抽样后的已售客票数据，存在发布滞后，也通常没有用户何时搜索或购买的完整信息。
- **80% 区间不是承诺。** 它表示在与校准数据近似同分布时的经验覆盖目标；节假日、突发事件、新航线或分布漂移都会导致覆盖不足。
- **准点概率不是航班状态。** 临时维护、机组、空管、天气和连锁延误可能在起飞前迅速变化。
- **美国数据覆盖有限。** BTS 数据主要服务于美国国内航空统计；示例中的非美国机场代码不意味着模型已经具备相应地区的真实训练覆盖。
- 不应将输出用于自动购买、拒绝退款、差别定价、安全关键决策，或在没有人工复核的情况下作出重大财务决定。

更完整的适用范围、评估要求与风险说明见 [模型卡](MODEL_CARD.md)。

## 许可证

代码以 [MIT License](LICENSE) 发布。BTS 与 NOAA 数据不随本仓库再分发；下载和使用外部数据时，请自行核对其最新条款、引用要求和访问限制。
