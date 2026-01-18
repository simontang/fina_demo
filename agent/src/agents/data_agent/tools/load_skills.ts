/**
 * Load Skills Tools
 * Tools for loading skill metadata and content
 */

import z from "zod";
import { registerToolLattice } from "@axiom-lattice/core";
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import * as analystSkill from "../skills/analyst";
import * as dataVisualizationSkill from "../skills/data-visualization";
import * as sqlQuerySkill from "../skills/sql-query";
import * as analysisMethodologySkill from "../skills/analysis-methodology";
import * as notebookReportSkill from "../skills/notebook-report";
import * as infographicCreatorSkill from "../skills/infographic-creator";

// Type definition for skill structure
interface Skill {
  name: string;
  description: string;
  prompt: string;
}

// Registry of all available skills
const skillsRegistry: Record<string, Skill> = {
  analyst: analystSkill.analyst,
  "data-visualization": dataVisualizationSkill.dataVisualization,
  "sql-query": sqlQuerySkill.sqlQuery,
  "analysis-methodology": analysisMethodologySkill.analysisMethodology,
  "notebook-report": notebookReportSkill.notebookReport,
  "infographic-creator": infographicCreatorSkill.infographicCreator,
};

/**
 * Load all skills and return their metadata (name and description, without prompt)
 */
registerToolLattice(
  "load_skills",
  {
    name: "load_skills",
    description:
      "Load all available skills and return their metadata (name and description). This tool returns skill information without the prompt content. Use this to discover what skills are available.",
    needUserApprove: false,
    schema: z.object({}),
  },
  async (_input: Record<string, never>, _config: LangGraphRunnableConfig) => {
    try {
      const skillsMeta = Object.values(skillsRegistry).map((skill) => ({
        name: skill.name,
        description: skill.description,
      }));

      return JSON.stringify(skillsMeta, null, 2);
    } catch (error) {
      return `Error loading skills: ${error instanceof Error ? error.message : String(error)
        }`;
    }
  }
);

/**
 * Load a specific skill's content and return its prompt
 */
registerToolLattice(
  "load_skill_content",
  {
    name: "load_skill_content",
    description:
      "Load a specific skill's content by name and return its prompt. Use this tool to get the full prompt content for a skill that you want to use.",
    needUserApprove: false,
    schema: z.object({
      skill_name: z.string().describe("The name of the skill to load"),
    }),
  },
  async (input: { skill_name: string }, _config: LangGraphRunnableConfig) => {
    try {
      const skill = skillsRegistry[input.skill_name];

      if (!skill) {
        const availableSkills = Object.keys(skillsRegistry).join(", ");
        return `Skill "${input.skill_name}" not found. Available skills: ${availableSkills}`;
      }

      return skill.prompt;
    } catch (error) {
      return `Error loading skill content: ${error instanceof Error ? error.message : String(error)
        }`;
    }
  }
);
