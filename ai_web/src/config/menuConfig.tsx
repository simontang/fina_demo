import {
  RocketOutlined,
  UserSwitchOutlined,
} from "@ant-design/icons";
import type { SideMenuItemConfig } from "@axiom-lattice/react-sdk";

/**
 * Dynamic workspace menu items — these are shown conditionally
 * based on the assistants available to the current tenant.
 */

export interface DynamicMenuGroup {
  /** Menu group name (displayed as section header in the sidebar) */
  group: string;
  /** Menu items belonging to this group */
  items: SideMenuItemConfig[];
}

/** All known dynamic menu groups and their items */
export const DYNAMIC_MENU_GROUPS: DynamicMenuGroup[] = [
  {
    group: "Orchestrators",
    items: [
      {
        id: "task_agent_dormant_reactivation",
        type: "route" as const,
        name: "Dormant Reactivation",
        icon: <RocketOutlined />,
        order: -2,
        group: "Orchestrators",
      },
    ],
  },
  {
    group: "Task Agents",
    items: [
      {
        id: "task_agent_segment",
        type: "route" as const,
        name: "Segment Agent",
        icon: <UserSwitchOutlined />,
        order: -1,
        group: "Task Agents",
      },
    ],
  },
];

/**
 * Maps an assistant ID to the set of dynamic menu item IDs it supports.
 *
 * When an assistant ID is NOT listed here, none of the dynamic menu items
 * are shown.  Add new entries whenever a new assistant gains one or more
 * of the dynamic menu items above.
 */
export const ASSISTANT_MENU_MAP: Record<string, string[]> = {
  "dormant-reactivation": ["task_agent_dormant_reactivation"],
  "task-audience-discovery": ["task_agent_segment"],
};

/**
 * Return the dynamic {@link SideMenuItemConfig}s backed by the available
 * assistants.
 */
export function getActiveDynamicMenuItems(
  assistantIds: readonly string[],
): SideMenuItemConfig[] {
  const allowedIds = new Set(
    assistantIds.flatMap((assistantId) => ASSISTANT_MENU_MAP[assistantId] ?? []),
  );
  if (allowedIds.size === 0) return [];

  return DYNAMIC_MENU_GROUPS.flatMap((group) =>
    group.items.filter((item) => allowedIds.has(item.id)),
  );
}
