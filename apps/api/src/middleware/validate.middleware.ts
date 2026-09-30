import type { NextFunction, Request, Response } from "express";
import type { ZodSchema } from "zod";
import { ApiError } from "./error.middleware";

export function validateBody(schema: ZodSchema) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      next(new ApiError(400, "Invalid request body", result.error.flatten()));
      return;
    }
    req.body = result.data;
    next();
  };
}
