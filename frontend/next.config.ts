import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["@deck.gl/core", "@deck.gl/layers", "@deck.gl/react", "@deck.gl/aggregation-layers"],
};

export default nextConfig;
