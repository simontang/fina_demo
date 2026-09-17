export type CustomerTag = {
  tagId: string;
  name: string;
  dimension: string;
  evidence?: string;
};

type TagDef = { name: string; dimension: string };
type CustomerTagRef = { tagId: string; evidence?: string };

/**
 * Mock tag catalog. Each business tag has a stable 32-hex uuid + display name
 * + dimension. Replace with the real tag store later — the API contract stays.
 */
const TAGS: Record<string, TagDef> = {
  "9ce355bfacca49c4a9e9322a9317c196": { name: "抗老/紧致", dimension: "concerns" },
  "2f0a7d1c6b4e48a2b3c5d6e7f8091a2b": { name: "保湿", dimension: "concerns" },
  "4a1b2c3d5e6f47089a0b1c2d3e4f5061": { name: "黑钻光灿面霜", dimension: "interested_products" },
  "7c8d9e0f1a2b43c5d6e7f8091a2b3c4d": { name: "花精粹面霜", dimension: "interested_products" },
  "1e2f3a4b5c6d4789e0f1a2b3c4d5e6f7": { name: "小棕瓶系列", dimension: "interested_products" },
  "8a9b0c1d2e3f44567890a1b2c3d4e5f6": { name: "中", dimension: "purchase_intent" },
  "5d6e7f8091a24b3c4d5e6f708192a3b4": { name: "中", dimension: "price_sensitivity" },
  "c1d2e3f4a5b64c7d8e9f0a1b2c3d4e5f": { name: "情感关怀回访", dimension: "service_opportunities" },
  "0f1e2d3c4b5a46978899aabbccddeeff": {
    name: "健康关怀（腿、肩颈复健调理）",
    dimension: "service_opportunities",
  },
  "3b4c5d6e7f80491a2b3c4d5e6f708192": { name: "抖音互动维护", dimension: "service_opportunities" },
  "a1b2c3d4e5f6470899aabbccddeeff00": { name: "西安人", dimension: "custom_tags" },
  "6f5e4d3c2b1a40799887aabbccddeeff": { name: "雅诗兰黛高认可度", dimension: "custom_tags" },
  "11aa22bb33cc44dd55ee66ff77008899": { name: "抖音活跃", dimension: "custom_tags" },
  "0a1b2c3d4e5f46079887aabbccddeeff": { name: "公益/义工活动参与", dimension: "custom_tags" },
};

/** Mock customer → tag assignments (with optional evidence). */
const CUSTOMER_TAGS: Record<string, CustomerTagRef[]> = {
  cus_8899: [
    { tagId: "9ce355bfacca49c4a9e9322a9317c196", evidence: "很喜欢用黑钻光灿面霜" },
    { tagId: "2f0a7d1c6b4e48a2b3c5d6e7f8091a2b", evidence: "冬天选择花精粹面霜比较多" },
    { tagId: "4a1b2c3d5e6f47089a0b1c2d3e4f5061", evidence: "很喜欢用我们的黑钻光灿面霜" },
    { tagId: "7c8d9e0f1a2b43c5d6e7f8091a2b3c4d", evidence: "之前用过花精粹的面霜" },
    { tagId: "1e2f3a4b5c6d4789e0f1a2b3c4d5e6f7", evidence: "妹妹用小棕瓶系列" },
    { tagId: "8a9b0c1d2e3f44567890a1b2c3d4e5f6", evidence: "品牌认可度高，近期购买频率下降" },
    { tagId: "c1d2e3f4a5b64c7d8e9f0a1b2c3d4e5f", evidence: "父亲过世，处于情绪调整期" },
    { tagId: "0f1e2d3c4b5a46978899aabbccddeeff", evidence: "腿和肩颈一直在做调理" },
    { tagId: "3b4c5d6e7f80491a2b3c4d5e6f708192", evidence: "日常发在抖音" },
    { tagId: "a1b2c3d4e5f6470899aabbccddeeff00", evidence: "王女士，陕西西安人" },
    { tagId: "6f5e4d3c2b1a40799887aabbccddeeff", evidence: "对我们品牌的认可度还是很高的" },
    { tagId: "11aa22bb33cc44dd55ee66ff77008899", evidence: "他的日常也会发在抖音上面" },
    {
      tagId: "0a1b2c3d4e5f46079887aabbccddeeff",
      evidence: "参加公益活动、庙里义工活动",
    },
  ],
  cus_1001: [
    { tagId: "4a1b2c3d5e6f47089a0b1c2d3e4f5061" },
    { tagId: "1e2f3a4b5c6d4789e0f1a2b3c4d5e6f7", evidence: "长期复购" },
  ],
};

/** Look up a tag definition by id (name + dimension). */
export function getTag(tagId: string): TagDef | undefined {
  return TAGS[tagId];
}

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
