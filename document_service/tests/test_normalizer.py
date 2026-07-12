from app.engines.base import EngineResult
from app.normalizer import normalize_engine_result


def test_markdown_is_normalized_to_blocks_and_tables():
    result = EngineResult(
        engine="textin",
        raw={"data": {"markdown": "# 工作说明书\n\nScope paragraph\n\n| 项目 | 内容 |\n| --- | --- |\n| A | B |"}},
    )

    document = normalize_engine_result(source_asset_id="asset_1", result=result, raw_asset_id="asset_raw")

    assert document.engine == "textin"
    assert document.content["markdown"].startswith("# 工作说明书")
    assert document.content["plain_text"]
    assert [block.type for block in document.blocks] == ["heading", "paragraph", "table"]
    assert document.tables[0].rows == [["项目", "内容"], ["A", "B"]]
    assert document.raw_outputs["engine_result_asset_id"] == "asset_raw"


def test_nested_json_blocks_are_flattened_and_tables_extracted():
    result = EngineResult(
        engine="mineru",
        raw={"status": "completed"},
        json_content={
            "pages": [
                {
                    "page": 1,
                    "blocks": [
                        {"id": "title", "type": "heading", "text": "Statement of Work", "page": 1},
                        {
                            "id": "table1",
                            "type": "table",
                            "page": 2,
                            "rows": [
                                [{"text": "Role"}, {"text": "Owner"}],
                                [{"text": "PM"}, {"text": "Vendor"}],
                            ],
                        },
                    ],
                }
            ]
        },
    )

    document = normalize_engine_result(source_asset_id="asset_1", result=result, raw_asset_id="asset_raw")

    assert [block.id for block in document.blocks] == ["title", "table1"]
    assert document.tables[0].id == "t001"
    assert document.tables[0].rows == [["Role", "Owner"], ["PM", "Vendor"]]
    assert document.tables[0].page == 2


def test_markdown_link_and_emphasis_are_removed_from_plain_text():
    result = EngineResult(
        engine="datalab",
        raw={"markdown": "## Title\n\n[Link](https://example.com) and **bold** text."},
    )

    document = normalize_engine_result(source_asset_id="asset_1", result=result, raw_asset_id="asset_raw")

    assert "Link and bold text" in document.content["plain_text"]


def test_html_tables_in_markdown_are_extracted():
    result = EngineResult(
        engine="textin",
        raw={
            "markdown": (
                "# Title\n\n"
                '<table border="1"><tr><th>ID</th><th>Description</th></tr>'
                "<tr><td>1</td><td>Scope</td></tr></table>"
            )
        },
    )

    document = normalize_engine_result(source_asset_id="asset_1", result=result, raw_asset_id="asset_raw")

    assert document.tables[0].rows == [["ID", "Description"], ["1", "Scope"]]
