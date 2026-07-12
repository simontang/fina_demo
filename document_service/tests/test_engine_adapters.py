from app.config import Settings
from app.engines.mineru import MinerUAdapter
from app.engines.qwen_ocr import QwenOcrAdapter


def test_qwen_response_api_ocr_result_is_extracted():
    adapter = QwenOcrAdapter(Settings(_env_file=None, qwen_api_key="token"))
    raw = {
        "output": [
            {
                "content": [
                    {
                        "ocr_result": {
                            "markdown": "# Title\n\nBody",
                        }
                    }
                ]
            }
        ]
    }

    assert adapter._extract_response_content(raw) == "# Title\n\nBody"


def test_mineru_result_prefers_downloaded_markdown_and_json():
    adapter = MinerUAdapter(Settings(_env_file=None, mineru_token="token"))
    result = adapter._result(
        {
            "submitted": {"code": 0, "data": {"task_id": "task_1"}},
            "result": {
                "data": {"state": "done"},
                "downloaded": {
                    "markdown": "# MinerU",
                    "json": {"content_list.json": [{"type": "text", "text": "MinerU"}]},
                },
            },
        }
    )

    assert result.markdown == "# MinerU"
    assert result.json_content == {"content_list.json": [{"type": "text", "text": "MinerU"}]}
