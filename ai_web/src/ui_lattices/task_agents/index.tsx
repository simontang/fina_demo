import { FolderOpenFilled } from "@ant-design/icons";
import { regsiterElement } from "@axiom-lattice/react-sdk";
import type { TaskAgentTab } from "../../pages/task-agents/TaskAgentLayout";
import { TaskAgentLayout } from "../../pages/task-agents/TaskAgentLayout";
import {
  DormantReactivationArtifactCard,
  DormantReactivationArtifactPanel,
} from "./dormant-reactivation/DormantReactivationArtifact";
import { DormantReactivationLayout } from "./dormant-reactivation/DormantReactivationLayout";
import {
  ChurnDashboardCard,
  ChurnDashboardPanel,
  ChurnListCard,
  ChurnListPanel,
} from "./insights/ChurnViews";
import {
  NBADashboardCard,
  NBADashboardPanel,
  NBAListCard,
  NBAListPanel,
} from "./insights/ProductNbaViews";
import { SegmentArtifactCard, SegmentArtifactPanel } from "./segment/SegmentArtifact";

regsiterElement("segment_artifact_workbench", {
  card_view: SegmentArtifactCard,
  side_app_view: SegmentArtifactPanel,
});

regsiterElement("churn_scoring_list", {
  card_view: ChurnListCard,
  side_app_view: ChurnListPanel,
});

regsiterElement("churn_scoring_dashboard", {
  card_view: ChurnDashboardCard,
  side_app_view: ChurnDashboardPanel,
});

regsiterElement("product_nba_list", {
  card_view: NBAListCard,
  side_app_view: NBAListPanel,
});

regsiterElement("product_nba_dashboard", {
  card_view: NBADashboardCard,
  side_app_view: NBADashboardPanel,
});

regsiterElement("dormant_reactivation_artifact_workbench", {
  card_view: DormantReactivationArtifactCard,
  side_app_view: DormantReactivationArtifactPanel,
});

const segmentTabs: TaskAgentTab[] = [
  {
    key: "artifact",
    label: "Artifact",
    icon: <FolderOpenFilled />,
    componentKey: "segment_artifact_workbench",
  },
];

regsiterElement("task_agent_segment", {
  card_view: () => null,
  side_app_view: () => (
    <TaskAgentLayout assistantId="task-audience-discovery" tabs={segmentTabs} />
  ),
});

const dormantReactivationTabs: TaskAgentTab[] = [
  {
    key: "artifact",
    label: "Artifact",
    icon: <FolderOpenFilled />,
    componentKey: "dormant_reactivation_artifact_workbench",
  },
];

regsiterElement("task_agent_dormant_reactivation", {
  card_view: () => null,
  side_app_view: () => (
    <DormantReactivationLayout assistantId="dormant-reactivation" tabs={dormantReactivationTabs} />
  ),
});
