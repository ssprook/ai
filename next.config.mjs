/** @type {import('next').NextConfig} */
const nextConfig = {
  // output: 'export',
  distDir: 'out',
  images: {
    unoptimized: true,
    domains: [
      "source.unsplash.com",
      "images.unsplash.com",
      "ext.same-assets.com",
      "ugc.same-assets.com",
    ],
    remotePatterns: [
      {
        protocol: "https",
        hostname: "source.unsplash.com",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "images.unsplash.com",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "ext.same-assets.com",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "ugc.same-assets.com",
        pathname: "/**",
      },
    ],
  },
};

export default nextConfig;
// /** @type {import('next').NextConfig} */
// const nextConfig = {
//   output: 'export',
//   distDir: 'out',
//   eslint: {
//     // Ignore les erreurs ESLint pendant la build de production
//     ignoreDuringBuilds: true,
//   },
//   images: {
//     unoptimized: true,
//     domains: [
//       "source.unsplash.com",
//       "images.unsplash.com",
//       "ext.same-assets.com",
//       "ugc.same-assets.com",
//     ],
//     remotePatterns: [
//       {
//         protocol: "https",
//         hostname: "source.unsplash.com",
//         pathname: "/**",
//       },
//       {
//         protocol: "https",
//         hostname: "images.unsplash.com",
//         pathname: "/**",
//       },
//       {
//         protocol: "https",
//         hostname: "ext.same-assets.com",
//         pathname: "/**",
//       },
//       {
//         protocol: "https",
//         hostname: "ugc.same-assets.com",
//         pathname: "/**",
//       },
//     ],
//   },
// };

// export default nextConfig;
