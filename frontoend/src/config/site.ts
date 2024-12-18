export interface NavItem {
    title: string;
    href: string;
    icon?: string;
    disabled?: boolean;
    external?: boolean;
    label?: string;
  }

  export interface NavSection {
    title: string;
    items: NavItem[];
  }

  export interface AuthConfig {
    loginPath: string;
    registerPath: string;
    forgotPasswordPath: string;
    callbackUrl: string;
  }

  export interface ApiConfig {
    baseUrl: string;
    endpoints: {
      dashboard: string;
      graphs: string;
      analysis: string;
      health: string;
    };
  }

  export interface SiteConfig {
    name: string;
    description: string;
    url: string;
    ogImage: string;
    links: {
      github: string;
      docs: string;
    };
    creator: string;
    auth: AuthConfig;
    api: ApiConfig;
    nav: {
      main: NavSection[];
      dashboard: NavItem[];
    };
  }

  export const siteConfig: SiteConfig = {
    name: "ダッシュボード分析システム",
    description: "データ分析とビジュアライゼーションのための統合プラットフォーム",
    url: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
    ogImage: "/og.png",
    links: {
      github: "https://github.com/your-repo/dashboard-system",
      docs: "/docs",
    },
    creator: "Your Company Name",
    auth: {
      loginPath: "/auth/login",
      registerPath: "/auth/register",
      forgotPasswordPath: "/auth/forgot-password",
      callbackUrl: "/dashboard",
    },
    api: {
      baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
      endpoints: {
        dashboard: "/dashboard",
        graphs: "/graphs",
        analysis: "/analysis",
        health: "/health",
      },
    },
    nav: {
      main: [
        {
          title: "メインメニュー",
          items: [
            {
              title: "ダッシュボード",
              href: "/dashboard",
              icon: "layout-dashboard",
            },
            {
              title: "グラフ",
              href: "/graphs",
              icon: "line-chart",
            },
            {
              title: "分析",
              href: "/analysis",
              icon: "bar-chart",
            },
          ],
        },
        {
          title: "設定",
          items: [
            {
              title: "プロフィール",
              href: "/profile",
              icon: "user",
            },
            {
              title: "設定",
              href: "/settings",
              icon: "settings",
            },
          ],
        },
      ],
      dashboard: [
        {
          title: "概要",
          href: "/dashboard",
        },
        {
          title: "データ入力",
          href: "/dashboard/input",
        },
        {
          title: "レポート",
          href: "/dashboard/reports",
        },
      ],
    },
  };