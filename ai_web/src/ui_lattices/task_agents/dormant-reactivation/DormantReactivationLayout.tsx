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
import type {
  SegmentDataPage,
  SegmentDataVO,
  SegmentDefinitionVO,
} from "../segment/types";

interface DormantReactivationLayoutProps {
  assistantId: string;
  tabs: TaskAgentTab[];
}

interface SegmentArtifact {
  definition: SegmentDefinitionVO;
  data: SegmentDataVO;
  option: string;
}

type ArtifactLoadStatus = "loading" | "ready" | "empty" | "error";

function formatCreatedAt(value: string): string {
  return value?.slice(0, 16).replace("T", " ") || "unknown time";
}

function toSegmentArtifact(
  definition: SegmentDefinitionVO,
  data: SegmentDataVO,
): SegmentArtifact {
  return {
    definition,
    data,
    option: `${definition.name} · ${data.rowCount.toLocaleString()} members · ${formatCreatedAt(data.createdAt)} · data #${data.id}`,
  };
}

function getExistingArtifactLabel(status: ArtifactLoadStatus): string {
  if (status === "loading") return "Use Existing Segment Artifact (Loading...)";
  if (status === "error") return "Use Existing Segment Artifact (Unavailable)";
  if (status === "empty") return "Use Existing Segment Artifact (No Artifacts)";
  return "Use Existing Segment Artifact";
}

function buildQuickPrompts(
  artifacts: SegmentArtifact[],
  status: ArtifactLoadStatus,
): QuickPromptCategory[] {
  const artifactByOption = new Map(artifacts.map((artifact) => [artifact.option, artifact]));
  const options = artifacts.map((artifact) => artifact.option);
  const existingArtifactPrompt: QuickPromptItem = {
    key: "dormant_reactivation_existing_artifact",
    label: getExistingArtifactLabel(status),
    description: "Use the latest materialized result from an existing Segment.",
    icon: <FolderOpenOutlined />,
    disabled: status !== "ready",
    content: [
      {
        type: "text",
        value: "Start a Dormant Reactivation analysis using the existing Segment Artifact ",
      },
      {
        type: "select",
        key: "segment_artifact",
        props: {
          options,
          defaultValue: options[0],
          placeholder: "Select a Segment Artifact",
        },
        formatResult: (value: unknown) => {
          const artifact = artifactByOption.get(String(value));
          if (!artifact) return "[unavailable Segment Artifact]";
          const { definition, data } = artifact;
          return `"${definition.name}" (definitionId=${definition.id}, segmentDataId=${data.id}, runId=${data.runId}, rowCount=${data.rowCount})`;
        },
      },
      {
        type: "text",
        value:
          ". Treat this exact segment_data snapshot as the fixed main audience. Do not create, replace, or re-materialize the segment. First verify that the selected artifact still exists for the current tenant; if it is unavailable or deleted, stop and ask me to select another artifact instead of silently falling back. Then continue with churn diagnosis, offer, channel and content, approval, execution, and measurement planning.",
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
          label: "Rebuild Dormant Audience",
          description: "Create and materialize a new audience from current CDP data.",
          icon: <UsergroupAddOutlined />,
          content: [
            {
              type: "text",
              value:
                "Start a new Dormant Reactivation analysis using current CDP data. For this conversation, do not reuse any existing Segment Artifact. Create a new segment definition, materialize it into a new segment_data snapshot, and report definitionId, segmentDataId, runId, and rowCount before continuing with churn diagnosis, offer, channel and content, approval, execution, and measurement planning.",
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
  const [artifacts, setArtifacts] = useState<SegmentArtifact[]>([]);
  const [status, setStatus] = useState<ArtifactLoadStatus>("loading");

  useEffect(() => {
    let active = true;

    const loadArtifacts = async () => {
      setStatus("loading");
      try {
        const definitionResponse = await get<CdpApiResponse<SegmentDefinitionVO[]>>(
          `${CDP_API_BASE}/segment-definitions`,
        );
        const definitionData = unwrapCdpResponse(definitionResponse);
        const definitions = Array.isArray(definitionData) ? definitionData : [];
        const latestData = await Promise.all(
          definitions.map(async (definition) => {
            const response = await get<CdpApiResponse<SegmentDataPage>>(
              `${CDP_API_BASE}/segment-data?definitionId=${definition.id}&page=1&pageSize=1`,
            );
            const page = unwrapCdpResponse(response);
            const data = Array.isArray(page.items) ? page.items[0] : undefined;
            return data ? toSegmentArtifact(definition, data) : null;
          }),
        );

        if (!active) return;
        const availableArtifacts = latestData
          .filter((artifact): artifact is SegmentArtifact => artifact !== null)
          .sort((a, b) => (
            b.data.createdAt.localeCompare(a.data.createdAt) || b.data.id - a.data.id
          ));
        setArtifacts(availableArtifacts);
        setStatus(availableArtifacts.length > 0 ? "ready" : "empty");
      } catch {
        if (!active) return;
        setArtifacts([]);
        setStatus("error");
      }
    };

    void loadArtifacts();
    return () => {
      active = false;
    };
  }, [get]);

  const quickPrompts = useMemo(
    () => buildQuickPrompts(artifacts, status),
    [artifacts, status],
  );

  useEffect(() => {
    updateConfigValue("quickPromptsData", quickPrompts);
  }, [quickPrompts, updateConfigValue]);

  useEffect(() => () => {
    updateConfigValue("quickPromptsData", previousQuickPromptsRef.current);
  }, [updateConfigValue]);

  return <TaskAgentLayout assistantId={assistantId} tabs={tabs} />;
};
