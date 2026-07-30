/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    // Allow Openverse image thumbnails from any host
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
};

module.exports = nextConfig;
