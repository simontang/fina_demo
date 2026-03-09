import React from "react";

export interface LogoProps {
    width?: number;
    height?: number;
    className?: string;
}

export const Logo: React.FC<LogoProps> = ({
    width = 48,
    height = 48,
    className = "",
}) => {
    return (
        <div
            style={{
                width: 64,
                height: 64,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: 8,
                boxSizing: "border-box",
                margin: "0 auto",
            }}
        >
            <img
                src="./logo.png"
                alt="Logo"
                width={width}
                height={height}
                className={className}
                style={{ objectFit: "contain" }}
            />
        </div>
    );
};

export default Logo;
