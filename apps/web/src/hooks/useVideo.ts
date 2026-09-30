import { useMutation } from "@tanstack/react-query";
import { videoApi } from "../services/api";

export function useUploadVideo() {
  return useMutation({ mutationFn: (file: File) => videoApi.upload(file) });
}
