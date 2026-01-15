import type { AuthProvider, DataProvider } from "@refinedev/core";
import { notification } from "antd";

export const TOKEN_KEY = "refine-auth";
export const USER_KEY = "refine-user";
const apiUrl = import.meta.env.VITE_API_URL;
export const authProvider: AuthProvider = {
  login: async ({ email, password }) => {
    try {
      // 调用新的 admin login API
      const response = await fetch(`${apiUrl}/login`, {
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
        window.location.href = "/login";
        throw new Error("Authentication failed");
      }

      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP error! status: ${response.status}`
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

      const url = `${baseUrl}/${resource}?${params.toString()}`;

      try {
        const response = await fetch(url, {
          method: "GET",
          headers: getHeaders(),
        });

        const data = await handleResponse(response);

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
      const url = `${baseUrl}/${resource}/${id}`;

      try {
        const response = await fetch(url, {
          method: "GET",
          headers: getHeaders(),
        });

        const data = await handleResponse(response);

        return {
          data: data.data,
        };
      } catch (error) {
        console.error(`Error fetching ${resource} with id ${id}:`, error);
        throw error;
      }
    },

    create: async ({ resource, variables }) => {
      const url = `${baseUrl}/${resource}`;

      try {
        const response = await fetch(url, {
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
      const url = `${baseUrl}/${resource}/${id}`;

      try {
        const response = await fetch(url, {
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
      const url = `${baseUrl}/${resource}/${id}`;

      try {
        const response = await fetch(url, {
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
      const fullUrl = url.startsWith("http") ? url : `${baseUrl}${url}`;

      try {
        const response = await fetch(fullUrl, {
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
