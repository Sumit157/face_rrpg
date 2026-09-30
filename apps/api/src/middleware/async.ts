import type { NextFunction, Request, RequestHandler, Response } from "express";

/** Wraps an async Express handler so thrown/rejected errors reach
 * errorHandler instead of crashing the process (Express 4 limitation). */
export function asyncHandler(fn: RequestHandler): RequestHandler {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}
