# Codex 任务说明：糖果毛利报告 JE 口径统一 + 接口与环境

本文档供在 Codex 中执行「董事会级别汇报-糖果毛利」相关开发与数据核对时使用，包含：任务目标、接口调用方式、SSH 端口转发、需阅读的 meta 与口径定义。

---

## 一、任务目标

1. **文档**：`metrics-server/docs/董事会级别汇报-糖果毛利.md`
   - 将报告中**毛利（及 COGS）计算口径统一为 MTC_VW_AI_JournalEntry**。
   - **收入**：科目代码以 `4` 开头的行，净额 = `Credit - Debit`（本位币）。
   - **商品成本（COGS）**：科目代码以 `5` 开头的行，净额 = `Credit - Debit`（通常为负）。
   - **毛利率** = (收入 + COGS) / 收入 × 100%（COGS 为负，故分子为收入加 COGS）。
   - 糖果品类：收入科目 `401020100`，成本科目 `501030100`。
   - **去掉**所有基于 STOCK.AvgPrice、OINV 售价对比、或“单位成本×数量”估算毛利的表述与结论；不得再出现“77–99% SKU 毛利”“成本评估”等以非 JE 口径为依据的结论。

2. **一致性**：若报告中有按渠道（CC1/CC2）、按年度等拆分毛利，均需注明数据来自 **JournalEntry** 的 `PrcName`/`PrcName2`（及 `RefDate`），并与 `metrics-server/docs/毛利率分析V1.0-0303.md` 中的 JE 口径保持一致。

---

## 二、如何调用 metrics-server 接口

- **服务**：本地 metrics-server，默认端口 **5704**（以项目实际配置为准）。
- **数据源**：datasourceId = **3**，对应 HANA 库 `ZZZZ_KAIMAYMTC_TEST`（通过 SSH 转发访问，见第三节）。

**请求示例**

- **方式**：`POST`  
- **URL**：`http://localhost:5704/api/v1/metrics/query`  
- **Body（JSON）**：
  ```json
  {
    "datasourceId": 3,
    "customSql": "<HANA SQL 语句>",
    "params": {}
  }
  ```
- **SQL 注意**：HANA 列名区分大小写，建议用**双引号**标识符，例如：`"RefDate"`, `"Account"`, `"Credit"`, `"Debit"`, `"ProfitCode"`, `"PrcName"`, `"OcrCode2"`, `"PrcName2"`。
- **Schema**：当前 datasource 3 的 `currentSchema` 一般为 `ZZZZ_KAIMAYMTC_TEST`，视图名为 `MTC_VW_AI_JOURNAL_ENTRY`（或项目内实际表/视图名，需与 meta 一致）。

**糖果毛利 JE 汇总示例（2025 年）**

```sql
SELECT
  SUM(CASE WHEN "Account" LIKE '4%' THEN "Credit" - "Debit" ELSE 0 END) AS revenue,
  SUM(CASE WHEN "Account" LIKE '5%' THEN "Credit" - "Debit" ELSE 0 END) AS cogs
FROM "ZZZZ_KAIMAYMTC_TEST"."MTC_VW_AI_JOURNAL_ENTRY"
WHERE YEAR("RefDate") = 2025
```

按渠道（CC2）拆分示例：

```sql
SELECT
  "PrcName2" AS channel,
  SUM(CASE WHEN "Account" LIKE '4%' THEN "Credit" - "Debit" ELSE 0 END) AS revenue,
  SUM(CASE WHEN "Account" LIKE '5%' THEN "Credit" - "Debit" ELSE 0 END) AS cogs
FROM "ZZZZ_KAIMAYMTC_TEST"."MTC_VW_AI_JOURNAL_ENTRY"
WHERE YEAR("RefDate") = 2025
GROUP BY "PrcName2"
```

（若表/视图在 default schema 下，可省略 schema 前缀，以实际库结构为准。）

---

## 三、SSH 端口转发（访问 HANA）

- **HANA 地址**：`8.219.185.18:35015`（仅内网/跳板可访问）。
- **跳板机**：`deploy@demo.alphafina.cn`（以实际账号为准）。
- **本地转发**：将本地 `35015` 映射到 HANA `8.219.185.18:35015`，metrics-server 配置为连接 `localhost:35015`。

**命令（在本地终端执行，保持运行）**：

```bash
ssh -N -L 35015:8.219.185.18:35015 deploy@demo.alphafina.cn
```

- **含义**：`-L 35015:8.219.185.18:35015` 表示本机 35015 → 跳板机 → 8.219.185.18:35015。
- **注意**：该窗口需保持开启；关闭后本地 35015 无法访问 HANA。metrics-server 的 datasource 3 应配置为 `jdbc:sap://localhost:35015?currentSchema=ZZZZ_KAIMAYMTC_TEST`（或等价配置）。

---

## 四、需要阅读的 meta 与资源

1. **JournalEntry 视图字段与口径**
   - **文件**：`metrics-server/src/main/resources/meta/MTC_VW_AI_JOURNAL_ENTRY.csv`
   - **用途**：确认 HANA 中列名（如 `RefDate`, `Account`, `Debit`, `Credit`, `ProfitCode`, `PrcName`, `OcrCode2`, `PrcName2`）与类型，以及本报告所用“收入=4xx、COGS=5xx”的科目规则。

2. **其他 MTC 视图（若报告涉及 OINV、STOCK 等）**
   - 目录：`metrics-server/src/main/resources/meta/`
   - 常见文件：`MTC_VW_AI_OINV.csv`、`MTC_VW_AI_ODLN.csv`、`MTC_VW_AI_OPDN.csv` 等；本任务**毛利口径仅依赖 JournalEntry**，其余视图仅作维度/校验参考。

3. **口径与科目映射参考**
   - **文件**：`metrics-server/docs/毛利率分析V1.0-0303.md`
   - **用途**：与董事会报告 v3.0 的 JE 口径、科目映射（如 401020100/501030100）保持一致。

---

## 五、本地运行 metrics-server（可选）

- **位置**：`metrics-server/`（Gradle 项目）。
- **启动**：在项目根目录执行  
  `./gradlew bootRun`  
  或使用 IDE 运行 Spring Boot 主类。
- **确认**：datasource 3 的 JDBC URL 指向 `localhost:35015`，且 SSH 转发已建立后再调用 `POST /api/v1/metrics/query`。

---

## 六、检查清单（在 Codex 中完成本任务时）

- [ ] 报告中所有“毛利”“COGS”“毛利率”均标明来自 **MTC_VW_AI_JournalEntry**（4xx/5xx）。
- [ ] 已删除或重写所有基于 STOCK.AvgPrice、OINV 售价对比、单位成本估算的结论与表格。
- [ ] 糖果科目 401020100（收入）、501030100（成本）与《毛利率分析V1.0-0303》一致。
- [ ] 按渠道/部门拆分处注明维度来自 JE 的 PrcName/PrcName2。
- [ ] 报告版本号更新为 v3.0，并在文末注明“毛利/COGS 口径：MTC_VW_AI_JournalEntry”。

如需在 Codex 中直接跑数验证，可按第二节构造 `customSql`，用第三节保持 SSH 转发，结合第四节 meta 核对字段与口径。
