import type { ThemeConfig } from "antd";
import { axiomAntdTheme } from "@axiom-lattice/react-sdk";

/**
 * Adjust color brightness by percentage
 * Positive percent = lighter, Negative percent = darker
 */
export function adjustColor(hex: string, percent: number): string {
    const num = parseInt(hex.replace("#", ""), 16);
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = ((num >> 8) & 0x00ff) + amt;
    const B = (num & 0x0000ff) + amt;
    return (
        "#" +
        (
            0x1000000 +
            (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 +
            (G < 255 ? (G < 1 ? 0 : G) : 255) * 0x100 +
            (B < 255 ? (B < 1 ? 0 : B) : 255)
        )
            .toString(16)
            .slice(1)
    );
}

/**
 * Generate complete Ant Design theme based on primary color
 * Automatically calculates hover, active, and background colors
 */
export function generateTheme(primaryColor: string): ThemeConfig {
    return {
        ...axiomAntdTheme,
        token: {
            ...axiomAntdTheme.token,
            colorPrimary: primaryColor,
            colorPrimaryHover: adjustColor(primaryColor, 20),
            colorPrimaryActive: adjustColor(primaryColor, -15),
            colorPrimaryBg: adjustColor(primaryColor, 45) + "20",
        },
    };
}

/**
 * Pre-defined brand colors
 */
export const brandColors = {
    purple: "#c574ff",
    blue: "#1890ff",
    green: "#52c41a",
    orange: "#fa8c16",
    red: "#f5222d",
    pink: "#eb2f96",
    cyan: "#13c2c2",
    geekblue: "#2f4554",
    lime: "#a0d911",
    gold: "#faad14",
} as const;

/**
 * Create theme with brand color
 */
export function createBrandTheme(
    colorKey: keyof typeof brandColors
): ThemeConfig {
    return generateTheme(brandColors[colorKey]);
}
