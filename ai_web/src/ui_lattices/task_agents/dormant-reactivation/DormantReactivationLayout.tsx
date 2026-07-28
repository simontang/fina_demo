import React, { useEffect, useMemo, useRef, useState } from "react";
import { FolderOpenOutlined, UsergroupAddOutlined } from "@ant-design/icons";
import {
  type QuickPromptCategory,
  type QuickPromptItem,
  useApi,
  useLatticeChatShellContext,
} from "@axiom-lattice/react-sdk";
import {
  TaskAgentLayout,
  type TaskAgentTab,
} from "../../../pages/task-agents/TaskAgentLayout";
import { CDP_API_BASE, unwrapCdpResponse } from "../shared/cdp";
import type { CdpApiResponse } from "../shared/cdp";
import type { SegmentDefinitionVO } from "../segment/types";

interface DormantReactivationLayoutProps {
  assistantId: string;
  tabs: TaskAgentTab[];
}

type SegmentLoadStatus = "loading" | "ready" | "empty" | "error";

function formatSegmentOption(definition: SegmentDefinitionVO): string {
  return `${definition.name} (#${definition.id})`;
}

function getExistingArtifactLabel(status: SegmentLoadStatus): string {
  if (status === "loading") return "Select Existing Segment (Loading...)";
  if (status === "error") return "Select Existing Segment (Unavailable)";
  if (status === "empty") return "Select Existing Segment (No Segments)";
  return "Select Existing Segment";
}

function buildQuickPrompts(
  definitions: SegmentDefinitionVO[],
  status: SegmentLoadStatus,
): QuickPromptCategory[] {
  const definitionByOption = new Map(
    definitions.map((definition) => [formatSegmentOption(definition), definition]),
  );
  const options = Array.from(definitionByOption.keys());
  const existingArtifactPrompt: QuickPromptItem = {
    key: "dormant_reactivation_existing_artifact",
    label: getExistingArtifactLabel(status),
    description: "Select an existing Segment.",
    icon: <FolderOpenOutlined />,
    content: [
      {
        type: "text",
        value: "Run Dormant Reactivation using this existing Segment: ",
      },
      {
        type: "select",
        key: "segment_artifact",
        props: {
          options,
          defaultValue: options[0],
          placeholder: "Select an Existing Segment",
        },
        formatResult: (value: unknown) => {
          const definition = definitionByOption.get(String(value));
          if (!definition) return "[unavailable Segment]";
          return `"${definition.name}" (definitionId=${definition.id})`;
        },
      },
      {
        type: "text",
        value:
          ".",
      },
    ],
  };

  return [
    {
      key: "dormant_reactivation",
      title: "Dormant Reactivation",
      items: [
        {
          key: "dormant_reactivation_rebuild_audience",
          label: "Describe Target Audience",
          description: "Describe the dormant audience in natural language and create a new Segment.",
          icon: <UsergroupAddOutlined />,
          content: [
            {
              type: "text",
              value: "Run Dormant Reactivation for this audience: ",
            },
            {
              type: "input",
              key: "audience_description",
              props: {
                placeholder: "e.g. high-value members with no purchase in the last 120 days",
              },
            },
            {
              type: "text",
              value: ".",
            },
          ],
        },
        existingArtifactPrompt,
      ],
    },
  ];
}

export const DormantReactivationLayout: React.FC<DormantReactivationLayoutProps> = ({
  assistantId,
  tabs,
}) => {
  const { get } = useApi();
  const { config, updateConfigValue } = useLatticeChatShellContext();
  const previousQuickPromptsRef = useRef(config.quickPromptsData);
  const [definitions, setDefinitions] = useState<SegmentDefinitionVO[]>([]);
  const [status, setStatus] = useState<SegmentLoadStatus>("loading");

  useEffect(() => {
    let active = true;

    const loadDefinitions = async () => {
      setStatus("loading");
      try {
        const definitionResponse = await get<CdpApiResponse<SegmentDefinitionVO[]>>(
          `${CDP_API_BASE}/segment-definitions`,
        );
        const definitionData = unwrapCdpResponse(definitionResponse);
        const loadedDefinitions = Array.isArray(definitionData) ? definitionData : [];

        if (!active) return;
        setDefinitions(loadedDefinitions);
        setStatus(loadedDefinitions.length > 0 ? "ready" : "empty");
      } catch {
        if (!active) return;
        setDefinitions([]);
        setStatus("error");
      }
    };

    void loadDefinitions();
    return () => {
      active = false;
    };
  }, [get]);

  const quickPrompts = useMemo(
    () => buildQuickPrompts(definitions, status),
    [definitions, status],
  );

  useEffect(() => {
    updateConfigValue("quickPromptsData", quickPrompts);
  }, [quickPrompts, updateConfigValue]);

  useEffect(() => () => {
    updateConfigValue("quickPromptsData", previousQuickPromptsRef.current);
  }, [updateConfigValue]);

  return <TaskAgentLayout assistantId={assistantId} tabs={tabs} />;
};
