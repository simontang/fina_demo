// Session storage utilities for authentication
const TOKEN_KEY = "axiom-auth-token";
const USER_KEY = "axiom-user";
const TENANT_KEY = "axiom-current-tenant";

export const getToken = (): string | null => {
  return sessionStorage.getItem(TOKEN_KEY);
};

export const setToken = (token: string): void => {
  sessionStorage.setItem(TOKEN_KEY, token);
};

export const getUser = (): any | null => {
  const userStr = sessionStorage.getItem(USER_KEY);
  return userStr ? JSON.parse(userStr) : null;
};

export const setUser = (user: any): void => {
  sessionStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const getCurrentTenant = (): any | null => {
  const tenantStr = sessionStorage.getItem(TENANT_KEY);
  return tenantStr ? JSON.parse(tenantStr) : null;
};

export const setCurrentTenant = (tenant: any): void => {
  sessionStorage.setItem(TENANT_KEY, JSON.stringify(tenant));
};

export const clearAuth = (): void => {
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
  sessionStorage.removeItem(TENANT_KEY);
};
