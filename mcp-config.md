# Horizon MCP Server 配置

## 启动命令

```bash
cd /Users/kk233/Horizon && uv run python -m src.mcp.server
```

## Cursor 配置

在 `~/.cursor/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "horizon": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "/Users/kk233/Horizon"
    }
  }
}
```

## 可用 Tools

- `hz_validate_config` - 验证配置
- `hz_fetch_items` - 抓取内容
- `hz_score_items` - AI 评分
- `hz_filter_items` - 过滤低分内容
- `hz_enrich_items` - 丰富摘要
- `hz_generate_summary` - 生成日报
- `hz_run_pipeline` - 一键运行完整流程
- `hz_list_runs` - 查看运行历史
- `hz_get_run_meta` - 读取运行元数据
- `hz_get_run_stage` - 读取阶段数据
- `hz_get_run_summary` - 读取日报内容
- `hz_get_metrics` - 查看服务指标
- `hz_send_webhook` - 发送 Webhook 通知

## 可用 Resources

- `horizon://server/info` - 服务器信息
- `horizon://metrics` - 运行指标
- `horizon://runs` - 运行列表
- `horizon://runs/{run_id}/meta` - 运行元数据
- `horizon://runs/{run_id}/items/{stage}` - 阶段内容
- `horizon://runs/{run_id}/summary/{language}` - 日报
- `horizon://config/effective` - 生效配置
