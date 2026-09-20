import { BUSINESS_OBJECTS_BUILDER_PROMPT } from "../business_objects/prompt";
import { BUSINESS_OBJECTS_MODELING_SKILL } from "../business_objects/skill";

describe("business objects builder assets", () => {
  it("skill declares a version and all modeling sections", () => {
    expect(BUSINESS_OBJECTS_MODELING_SKILL.version).toBe("1.0.0");
    const content = BUSINESS_OBJECTS_MODELING_SKILL.content;
    expect(content).toContain("name: business-objects-modeling");
    for (const heading of [
      "## 1. 概念模型",
      "## 2. 命名规范",
      "## 3. store 选择与创建",
      "## 4. 字段类型映射",
      "## 5. 索引设计",
      "## 6. v1 演进约束",
      "## 7. 权限与 grant",
      "## 8. 安全与确认",
      "## 9. 标准工作流",
      "## 10. 验收清单",
    ]) {
      expect(content).toContain(heading);
    }
  });

  it("builder prompt requires loading the modeling skill first", () => {
    expect(BUSINESS_OBJECTS_BUILDER_PROMPT).toContain("CRITICAL FIRST ACTION");
    expect(BUSINESS_OBJECTS_BUILDER_PROMPT).toMatch(/skill_name:\s*"business-objects-modeling"/);
  });
});
