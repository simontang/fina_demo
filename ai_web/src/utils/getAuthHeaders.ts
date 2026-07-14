export function getAuthHeaders(): HeadersInit {
  const token = sessionStorage.getItem("lattice_token")
    || localStorage.getItem("axiom-auth-token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}
