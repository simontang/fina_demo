import React from "react";
import { useNavigate } from "react-router";
import { RegisterForm } from "@axiom-lattice/react-sdk";
import { AuthLogo, AuthTitle } from "./components";

export const RegisterPage: React.FC = () => {
  const navigate = useNavigate();

  const handleSuccess = () => {
    navigate("/login");
  };

  const handleCancel = () => {
    navigate("/login");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 420,
          padding: "40px 32px",
          background: "#fff",
          borderRadius: 12,
          boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
        }}
      >
        <AuthLogo />
        <AuthTitle />
        <RegisterForm
          onSuccess={handleSuccess}
          onCancel={handleCancel}
        />
      </div>
    </div>
  );
};
