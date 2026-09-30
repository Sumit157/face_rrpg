import http from "node:http";
import { createApp } from "./app";
import { config } from "./config";
import { logger } from "./config/logger";
import { initWebSocketGateway } from "./websocket/analysis.gateway";

const app = createApp();
const httpServer = http.createServer(app);

initWebSocketGateway(httpServer);

httpServer.listen(config.port, () => {
  logger.info(`rPPG API listening on port ${config.port} (${config.nodeEnv})`);
});
