import { PrismaClient } from "@prisma/client";

// A single shared Prisma client instance, per Prisma's recommended
// pattern for avoiding connection-pool exhaustion under hot-reload.
declare global {
  // eslint-disable-next-line no-var
  var __prisma: PrismaClient | undefined;
}

export const prisma = global.__prisma ?? new PrismaClient();

if ((process.env.NODE_ENV ?? "development") !== "production") {
  global.__prisma = prisma;
}
