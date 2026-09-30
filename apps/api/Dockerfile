FROM node:20-slim AS build
WORKDIR /app
RUN corepack enable
COPY pnpm-workspace.yaml pnpm-lock.yaml package.json tsconfig.base.json ./
COPY packages/shared-types ./packages/shared-types
COPY apps/api ./apps/api
COPY prisma ./prisma
RUN pnpm install --filter @rppg/api... --frozen-lockfile || pnpm install --filter @rppg/api...
RUN pnpm --filter @rppg/api run build
RUN npx prisma generate --schema ./prisma/schema.prisma
RUN pnpm --filter @rppg/api deploy --prod /deploy

FROM node:20-slim
WORKDIR /app
COPY --from=build /app/apps/api/dist ./dist
COPY --from=build /deploy/node_modules ./node_modules
COPY --from=build /deploy/package.json ./package.json
COPY prisma ./prisma
EXPOSE 3000
# Applies pending migrations (SQLite on the api_db volume) on boot.
CMD ["sh", "-c", "npx prisma migrate deploy --schema prisma/schema.prisma && node dist/server.js"]
