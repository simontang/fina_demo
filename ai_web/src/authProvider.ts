import type { AuthProvider, DataProvider } from "@refinedev/core";
import { notification } from "antd";

export const TOKEN_KEY = "refine-auth";
export const USER_KEY = "refine-user";
const apiUrl = import.meta.env.VITE_API_URL;

/** Python 预测/数据集 API 地址，默认同主机 8000 端口，可用 VITE_PYTHON_API_URL 覆盖 */
export function getPythonApiUrl(): string {
  const v = import.meta.env.VITE_PYTHON_API_URL;
  if (v && typeof v === "string") return v.replace(/\/$/, "");
  try {
    const u = new URL(apiUrl || "http://localhost:6203");
    u.port = "8000";
    u.pathname = "";
    u.search = "";
    u.hash = "";
    return u.origin;
  } catch {
    return "http://localhost:8000";
  }
}
export const authProvider: AuthProvider = {
  login: async ({ email, password }) => {
    try {
      // 调用登录 API（使用 /api/login 路径）
      const loginUrl = apiUrl && !apiUrl.startsWith("http") 
        ? `${apiUrl}/login` 
        : apiUrl 
        ? `${apiUrl}/api/login` 
        : "/api/login";
      const response = await fetch(loginUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: email, // 使用 email 作为用户名
          password: password,
        }),
      });

      const result = await response.json();

      if (result.success && result.data) {
        // 保存 token 和用户信息
        localStorage.setItem(TOKEN_KEY, result.data.token);
        localStorage.setItem(USER_KEY, JSON.stringify(result.data.user));

        // enableAutoLogin();

        notification.success({
          message: "登录成功",
          description: `欢迎回来，${result.data.user.username || email}`,
        });

        return {
          success: true,
          redirectTo: "/",
        };
      } else {
        // 登录失败
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
    localStorage.removeItem(TOKEN_KEY);
    return {
      success: true,
      redirectTo: "/login",
    };
  },
  check: async () => {
    const token = localStorage.getItem(TOKEN_KEY);
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
  getPermissions: async () => null,
  getIdentity: async () => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) {
      return {
        id: 1,
        name: "John Doe",
        avatar: "https://i.pravatar.cc/300",
      };
    }
    return null;
  },
  onError: async (error) => {
    console.error(error);
    return { error };
  },
};

/**
 * Custom data provider with JWT authentication
 * 自定义数据提供者，自动添加 JWT 认证头
 */
export const createAuthenticatedDataProvider = (): DataProvider => {
  const baseUrl = apiUrl;
  const pythonApiUrl = getPythonApiUrl();
  const requestTimeoutMs = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? 30000);

  const fetchWithTimeout = async (
    url: string,
    init: RequestInit,
  ): Promise<Response> => {
    if (!requestTimeoutMs || requestTimeoutMs <= 0 || init.signal) {
      return fetch(url, init);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), requestTimeoutMs);
    try {
      return await fetch(url, { ...init, signal: controller.signal });
    } catch (err: unknown) {
      if ((err as any)?.name === "AbortError") {
        throw new Error(`Request timeout after ${requestTimeoutMs}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const getHeaders = (): Record<string, string> => {
    const token = localStorage.getItem(TOKEN_KEY);
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    return headers;
  };

  const handleResponse = async (response: Response) => {
    if (!response.ok) {
      if (response.status === 401) {
        // Token 过期或无效，清除本地存储并重定向到登录页
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        window.location.href = "/admin/login";
        throw new Error("Authentication failed");
      }

      const errorData = await response.json().catch(() => ({}));
      const message =
        (typeof (errorData as any)?.message === "string" && (errorData as any).message) ||
        (typeof (errorData as any)?.detail === "string" && (errorData as any).detail);
      throw new Error(
        message || `HTTP error! status: ${response.status}`
      );
    }

    return response.json();
  };

  return {
    getList: async ({ resource, pagination, filters, sorters }) => {
      const current = (pagination as any)?.currentPage || 1;
      const pageSize = (pagination as any)?.pageSize || 10;
      const params = new URLSearchParams();

      // 添加分页参数
      params.append("page", current.toString());
      params.append("size", pageSize.toString());

      // 添加筛选参数
      if (filters) {
        filters.forEach((filter: any) => {
          if (filter.value !== undefined && filter.value !== null) {
            params.append(filter.field, filter.value.toString());
          }
        });
      }

      // 添加排序参数
      if (sorters && sorters.length > 0) {
        const sortField = sorters[0].field;
        const sortOrder = sorters[0].order === "desc" ? "DESC" : "ASC";
        params.append("sortBy", sortField);
        params.append("sortOrder", sortOrder);
      }

      // 特殊处理 datasets 资源，使用 Python API 代理路径
      // 其他资源（如 agents, files）使用 /api/* 路径
      const resourceUrl =
        resource === "datasets"
          ? `/api/v1/datasets`
          : `/api/${resource}`;
      const url = `${resourceUrl}?${params.toString()}`;

      try {
        const response = await fetchWithTimeout(url, {
          method: "GET",
          headers: getHeaders(),
        });

        const data = await handleResponse(response);

        // 处理 datasets API 的响应格式
        if (resource === "datasets" && data.success) {
          return {
            data: data.data || [],
            total: data.total || data.data?.length || 0,
          };
        }

        return {
          data: data.data?.records || data.data || data.records || [],
          total: data.data?.total || data.total || 0,
        };
      } catch (error) {
        console.error(`Error fetching ${resource} list:`, error);
        throw error;
      }
    },

    getOne: async ({ resource, id }) => {
      // 特殊处理 datasets 资源，使用 Python API 代理路径
      // 其他资源（如 agents, files）使用 /api/* 路径
      const resourceUrl =
        resource === "datasets"
          ? `/api/v1/datasets/${id}`
          : `/api/${resource}/${id}`;

      try {
        const response = await fetchWithTimeout(resourceUrl, {
          method: "GET",
          headers: getHeaders(),
        });

        const data = await handleResponse(response);

        // 处理 datasets API 的响应格式
        if (resource === "datasets" && data.success) {
          return {
            data: data.data,
          };
        }

        return {
          data: data.data,
        };
      } catch (error) {
        console.error(`Error fetching ${resource} with id ${id}:`, error);
        throw error;
      }
    },

    create: async ({ resource, variables }) => {
      const url = `/api/${resource}`;

      try {
        const response = await fetchWithTimeout(url, {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(variables),
        });

        const data = await handleResponse(response);

        return {
          data: data.data,
        };
      } catch (error) {
        console.error(`Error creating ${resource}:`, error);
        throw error;
      }
    },

    update: async ({ resource, id, variables }) => {
      const url = `/api/${resource}/${id}`;

      try {
        const response = await fetchWithTimeout(url, {
          method: "PUT",
          headers: getHeaders(),
          body: JSON.stringify(variables),
        });

        const data = await handleResponse(response);

        return {
          data: data.data,
        };
      } catch (error) {
        console.error(`Error updating ${resource} with id ${id}:`, error);
        throw error;
      }
    },

    deleteOne: async ({ resource, id }) => {
      const url = `/api/${resource}/${id}`;

      try {
        const response = await fetchWithTimeout(url, {
          method: "DELETE",
          headers: getHeaders(),
        });

        await handleResponse(response);

        return {
          data: { id } as any,
        };
      } catch (error) {
        console.error(`Error deleting ${resource} with id ${id}:`, error);
        throw error;
      }
    },

    getApiUrl: () => baseUrl,

    // 自定义方法：批量更新状态
    custom: async ({ url, method, headers, meta }) => {
      // 如果 URL 已经是完整路径（以 http 开头），直接使用
      // 否则，如果以 /api 开头，直接使用；否则添加 /api 前缀
      const fullUrl = url.startsWith("http") 
        ? url 
        : url.startsWith("/api") 
        ? url 
        : `/api${url}`;

      try {
        const response = await fetchWithTimeout(fullUrl, {
          method: method || "GET",
          headers: {
            ...getHeaders(),
            ...headers,
          },
          body: meta?.body ? JSON.stringify(meta.body) : undefined,
        });

        const data = await handleResponse(response);

        return {
          data: data.data,
        };
      } catch (error) {
        console.error(`Error in custom request to ${url}:`, error);
        throw error;
      }
    },
  };
};
