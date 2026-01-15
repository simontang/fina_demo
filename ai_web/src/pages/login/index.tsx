import { AuthPage } from "@refinedev/antd";

export const Login = () => {
  return (
    <AuthPage
      type="login"
      title="FULI Agent Center"
      formProps={{
        initialValues: { email: "demo@fina.dev", password: "demodemo" },
      }}
    />
  );
};
