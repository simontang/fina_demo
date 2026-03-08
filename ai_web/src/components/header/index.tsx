import type { RefineThemedLayoutHeaderProps } from "@refinedev/antd";
import { useGetIdentity, useLogout } from "@refinedev/core";
import {
  Layout as AntdLayout,
  Avatar,
  Button,
  Space,
  Switch,
  theme,
  Typography,
  Dropdown,
  Menu,
  Tag,
} from "antd";
import React, { useContext } from "react";
import { useNavigate } from "react-router";
import { ExperimentOutlined, UserOutlined, LogoutOutlined, SwapOutlined } from "@ant-design/icons";
import { useAuth } from "@axiom-lattice/react-sdk";
import { ColorModeContext } from "../../contexts/color-mode";
import { clearAuth, getCurrentTenant } from "../../utils/sessionStorage";

const { Text } = Typography;
const { useToken } = theme;

type IUser = {
  id: number;
  name: string;
  avatar: string;
  email?: string;
};

export const Header: React.FC<RefineThemedLayoutHeaderProps> = ({
  sticky = true,
}) => {
  const { token } = useToken();
  const { data: user } = useGetIdentity<IUser>();
  const { mode, setMode } = useContext(ColorModeContext);
  const navigate = useNavigate();
  const { mutate: logout } = useLogout();
  const { tenants, currentTenant, user: authUser } = useAuth();

  const currentTenantData = getCurrentTenant();

  const handleLogout = () => {
    clearAuth();
    logout();
    navigate("/login");
  };

  const handleSwitchTenant = () => {
    clearAuth();
    // Keep token but clear tenant, then redirect to tenant select
    navigate("/tenant-select");
  };

  const userMenuItems = [
    {
      key: "profile",
      icon: <UserOutlined />,
      label: authUser?.email || user?.name || "用户",
      disabled: true,
    },
    {
      type: "divider" as const,
    },
    ...(tenants.length > 1
      ? [
          {
            key: "switch-tenant",
            icon: <SwapOutlined />,
            label: "切换租户",
            onClick: handleSwitchTenant,
          },
        ]
      : []),
    {
      key: "logout",
      icon: <LogoutOutlined />,
      label: "退出登录",
      onClick: handleLogout,
    },
  ];

  const headerStyles: React.CSSProperties = {
    backgroundColor: token.colorBgElevated,
    display: "flex",
    justifyContent: "flex-end",
    alignItems: "center",
    padding: "0px 24px",
    height: "64px",
    borderBottom: `1px solid ${token.colorBorder}`,
  };

  if (sticky) {
    headerStyles.position = "sticky";
    headerStyles.top = 0;
    headerStyles.zIndex = 100;
  }

  return (
    <AntdLayout.Header style={headerStyles}>
      <Space>
        {currentTenantData && (
          <Tag color="blue">{currentTenantData.name || "默认租户"}</Tag>
        )}
        <Button
          type="primary"
          icon={<ExperimentOutlined />}
          onClick={() => navigate("/workbench")}
        >
          Agent Studio
        </Button>
        <Switch
          checked={mode === "dark"}
          onChange={() => setMode(mode === "dark" ? "light" : "dark")}
          checkedChildren="🌛"
          unCheckedChildren="☀️"
        />
        <Dropdown
          menu={{ items: userMenuItems }}
          placement="bottomRight"
          arrow
        >
          <Space style={{ marginLeft: "8px", cursor: "pointer" }} size="middle">
            {user?.name && <Text strong>{user.name}</Text>}
            <Avatar
              src={user?.avatar}
              icon={!user?.avatar && <UserOutlined />}
              alt={user?.name}
            />
          </Space>
        </Dropdown>
      </Space>
    </AntdLayout.Header>
  );
};
