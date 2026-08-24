import type { AuthProvider } from "@refinedev/core";
import { notification } from "antd";
import { getToken, getUser, clearAuth, setToken, setUser } from "./utils/sessionStorage";

const apiUrl = import.meta.env.VITE_API_URL;

// Get base API path (use origin so non-standard ports like Electron's
// 127.0.0.1:5721 keep the port; standard 80/443 behave identically).
function getBaseAPIPath(): string {
  return window.location.origin;
}

const baseURL = apiUrl || getBaseAPIPath();

export const axiomAuthProvider: AuthProvider = {
  login: async ({ email, password }) => {
    try {
      const response = await fetch(`${baseURL}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        // Store in sessionStorage
        setToken(result.data.token);
        setUser(result.data.user);

        notification.success({
          message: "登录成功",
          description: `欢迎回来，${result.data.user.name || email}`,
        });

        return {
          success: true,
          redirectTo: "/agents/data",
        };
      } else {
        notification.error({
          message: "登录失败",
          description: result.message || "用户名或密码错误",
        });

        return {
          success: false,
          error: {
            message: result.message || "登录失败",
            name: "Invalid credentials",
          },
        };
      }
    } catch (error) {
      console.error("Login error:", error);

      notification.error({
        message: "网络错误",
        description: "无法连接到服务器，请检查网络连接",
      });

      return {
        success: false,
        error: {
          message: "Network error",
          name: "Connection failed",
        },
      };
    }
  },

  logout: async () => {
    try {
      const token = getToken();
      if (token) {
        await fetch(`${baseURL}/api/auth/logout`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`,
          },
        });
      }
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      clearAuth();
    }

    return {
      success: true,
      redirectTo: "/login",
    };
  },

  check: async () => {
    const token = getToken();
    if (token) {
      return {
        authenticated: true,
      };
    }

    return {
      authenticated: false,
      redirectTo: "/login",
    };
  },

  getPermissions: async () => {
    const user = getUser();
    if (user) {
      return user.role || null;
    }
    return null;
  },

  getIdentity: async () => {
    const user = getUser();
    if (user) {
      return {
        id: user.id,
        name: user.name || user.email,
        email: user.email,
        avatar: user.avatar,
      };
    }
    return null;
  },

  onError: async (error) => {
    console.error(error);
    return { error };
  },

  register: async ({ email, password, name }) => {
    try {
      const response = await fetch(`${baseURL}/api/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
          name,
        }),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        notification.success({
          message: "注册成功",
          description: "请使用新账号登录",
        });

        return {
          success: true,
          redirectTo: "/login",
        };
      } else {
        notification.error({
          message: "注册失败",
          description: result.message || "注册失败，请重试",
        });

        return {
          success: false,
          error: {
            message: result.message || "注册失败",
            name: "Registration failed",
          },
        };
      }
    } catch (error) {
      console.error("Register error:", error);

      notification.error({
        message: "网络错误",
        description: "无法连接到服务器，请检查网络连接",
      });

      return {
        success: false,
        error: {
          message: "Network error",
          name: "Connection failed",
        },
      };
    }
  },

  forgotPassword: async ({ email }) => {
    try {
      const response = await fetch(`${baseURL}/api/auth/forgot-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const result = await response.json();

      if (response.ok) {
        notification.success({
          message: "邮件已发送",
          description: "请检查您的邮箱以重置密码",
        });

        return {
          success: true,
        };
      } else {
        notification.error({
          message: "发送失败",
          description: result.message || "请重试",
        });

        return {
          success: false,
          error: {
            message: result.message || "发送失败",
            name: "Failed",
          },
        };
      }
    } catch (error) {
      console.error("Forgot password error:", error);

      return {
        success: false,
        error: {
          message: "Network error",
          name: "Connection failed",
        },
      };
    }
  },
};
