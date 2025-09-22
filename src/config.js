// src/config.js

const ENV = import.meta.env.MODE;

const config = {
  development: {
    API_BASE_URL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  },
  staging: {
    API_BASE_URL: import.meta.env.VITE_API_BASE_URL || "https://staging.api.yourapp.com",
  },
  production: {
    API_BASE_URL: import.meta.env.VITE_API_BASE_URL || "https://api.yourapp.com",
  },
};

export default config[ENV];

