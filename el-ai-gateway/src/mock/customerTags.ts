export type CustomerTag = {
  tagId: string;
  name: string;
  dimension: string;
  evidence?: string;
};

type TagDef = { name: string; dimension: string };
type CustomerTagRef = { tagId: string; evidence?: string };

/**
 * Mock tag catalog. Each business tag has a stable uuid + display name +
 * dimension. Replace with the real tag store later — the API contract stays.
 */
const TAGS: Record<string, TagDef> = {
  "b1f2c3d4-0001-4a01-8001-000000000001": { name: "抗老/紧致", dimension: "concerns" },
  "b1f2c3d4-0002-4a01-8001-000000000002": { name: "保湿", dimension: "concerns" },
  "b1f2c3d4-0003-4a01-8001-000000000003": { name: "黑钻光灿面霜", dimension: "interested_products" },
  "b1f2c3d4-0004-4a01-8001-000000000004": { name: "花精粹面霜", dimension: "interested_products" },
  "b1f2c3d4-0005-4a01-8001-000000000005": { name: "小棕瓶系列", dimension: "interested_products" },
  "b1f2c3d4-0006-4a01-8001-000000000006": { name: "中", dimension: "purchase_intent" },
  "b1f2c3d4-0007-4a01-8001-000000000007": { name: "中", dimension: "price_sensitivity" },
  "b1f2c3d4-0008-4a01-8001-000000000008": { name: "情感关怀回访", dimension: "service_opportunities" },
  "b1f2c3d4-0009-4a01-8001-000000000009": {
    name: "健康关怀（腿、肩颈复健调理）",
    dimension: "service_opportunities",
  },
  "b1f2c3d4-000a-4a01-8001-00000000000a": { name: "抖音互动维护", dimension: "service_opportunities" },
  "b1f2c3d4-000b-4a01-8001-00000000000b": { name: "西安人", dimension: "custom_tags" },
  "b1f2c3d4-000c-4a01-8001-00000000000c": { name: "雅诗兰黛高认可度", dimension: "custom_tags" },
  "b1f2c3d4-000d-4a01-8001-00000000000d": { name: "抖音活跃", dimension: "custom_tags" },
  "b1f2c3d4-000e-4a01-8001-00000000000e": { name: "公益/义工活动参与", dimension: "custom_tags" },
};

/** Mock customer → tag assignments (with optional evidence). */
const CUSTOMER_TAGS: Record<string, CustomerTagRef[]> = {
  cus_8899: [
    { tagId: "b1f2c3d4-0001-4a01-8001-000000000001", evidence: "很喜欢用黑钻光灿面霜" },
    { tagId: "b1f2c3d4-0002-4a01-8001-000000000002", evidence: "冬天选择花精粹面霜比较多" },
    { tagId: "b1f2c3d4-0003-4a01-8001-000000000003", evidence: "很喜欢用我们的黑钻光灿面霜" },
    { tagId: "b1f2c3d4-0004-4a01-8001-000000000004", evidence: "之前用过花精粹的面霜" },
    { tagId: "b1f2c3d4-0005-4a01-8001-000000000005", evidence: "妹妹用小棕瓶系列" },
    { tagId: "b1f2c3d4-0006-4a01-8001-000000000006", evidence: "品牌认可度高，近期购买频率下降" },
    { tagId: "b1f2c3d4-0008-4a01-8001-000000000008", evidence: "父亲过世，处于情绪调整期" },
    { tagId: "b1f2c3d4-0009-4a01-8001-000000000009", evidence: "腿和肩颈一直在做调理" },
    { tagId: "b1f2c3d4-000a-4a01-8001-00000000000a", evidence: "日常发在抖音" },
    { tagId: "b1f2c3d4-000b-4a01-8001-00000000000b", evidence: "王女士，陕西西安人" },
    { tagId: "b1f2c3d4-000c-4a01-8001-00000000000c", evidence: "对我们品牌的认可度还是很高的" },
    { tagId: "b1f2c3d4-000d-4a01-8001-00000000000d", evidence: "他的日常也会发在抖音上面" },
    {
      tagId: "b1f2c3d4-000e-4a01-8001-00000000000e",
      evidence: "参加公益活动、庙里义工活动",
    },
  ],
  cus_1001: [
    { tagId: "b1f2c3d4-0003-4a01-8001-000000000003" },
    { tagId: "b1f2c3d4-0005-4a01-8001-000000000005", evidence: "长期复购" },
  ],
};

/** Resolve a customer's tags. Returns undefined when the customer is unknown. */
export function getCustomerTags(customerId: string): CustomerTag[] | undefined {
  const refs = CUSTOMER_TAGS[customerId];
  if (!refs) return undefined;
  return refs.map((ref) => {
    const def = TAGS[ref.tagId];
    const tag: CustomerTag = { tagId: ref.tagId, name: def.name, dimension: def.dimension };
    if (ref.evidence) tag.evidence = ref.evidence;
    return tag;
  });
}
