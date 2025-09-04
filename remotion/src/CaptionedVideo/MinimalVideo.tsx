import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AbsoluteFill,
  CalculateMetadataFunction,
  cancelRender,
  continueRender,
  delayRender,
  getStaticFiles,
  OffthreadVideo,
  Sequence,
  useVideoConfig,
  watchStaticFile,
} from "remotion";
import { z } from "zod";
import { getVideoMetadata } from "@remotion/media-utils";
import { loadFont } from "../load-font";
import { NoCaptionFile } from "./NoCaptionFile";
import { Caption, createTikTokStyleCaptions } from "@remotion/captions";
import { MinimalStyle } from "./MinimalStyle";

export const minimalVideoSchema = z.object({
  src: z.string().optional(),
  videoPath: z.string().optional(),
  captionsPath: z.string().optional(),
});

export const calculateMinimalVideoMetadata: CalculateMetadataFunction<
  z.infer<typeof minimalVideoSchema>
> = async ({ props }) => {
  const fps = 30;
  // 使用默认时长，避免在metadata阶段访问视频文件
  const defaultDurationInSeconds = 10; // 10秒默认时长
  
  return {
    fps,
    durationInFrames: Math.floor(defaultDurationInSeconds * fps),
  };
};

const getFileExists = (file: string) => {
  const files = getStaticFiles();
  const fileExists = files.find((f) => {
    return f.src === file;
  });
  return Boolean(fileExists);
};

const SWITCH_CAPTIONS_EVERY_MS = 1200;

export const MinimalVideo: React.FC<{
  src?: string;
  videoPath?: string;
  captionsPath?: string;
}> = ({ src, videoPath, captionsPath }) => {
  const [subtitles, setSubtitles] = useState<Caption[]>([]);
  const [handle] = useState(() => delayRender());
  const { fps } = useVideoConfig();

  // 确定视频源和字幕文件路径
  const videoSrc = videoPath || src || "sample-video.mp4";
  const subtitlesFile = captionsPath || (src ? src.replace(/.mp4$/, ".json").replace(/.mkv$/, ".json").replace(/.mov$/, ".json").replace(/.webm$/, ".json") : "sample-video.json");

  const fetchSubtitles = useCallback(async () => {
    try {
      await loadFont();
      
      let data: Caption[];
      if (captionsPath) {
        // 如果提供了绝对路径，使用Node.js fs模块读取
        const fs = await import('fs/promises');
        const content = await fs.readFile(captionsPath, 'utf-8');
        data = JSON.parse(content) as Caption[];
      } else {
        // 否则使用默认的fetch方式
        const res = await fetch(subtitlesFile);
        data = (await res.json()) as Caption[];
      }
      
      setSubtitles(data);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [handle, subtitlesFile, captionsPath]);

  useEffect(() => {
    fetchSubtitles();

    if (!captionsPath) {
      // 只有在使用相对路径时才watch文件
      const c = watchStaticFile(subtitlesFile, () => {
        fetchSubtitles();
      });

      return () => {
        c.cancel();
      };
    }
  }, [fetchSubtitles, src, subtitlesFile, captionsPath]);

  const { pages } = useMemo(() => {
    return createTikTokStyleCaptions({
      combineTokensWithinMilliseconds: SWITCH_CAPTIONS_EVERY_MS,
      captions: subtitles ?? [],
    });
  }, [subtitles]);

  return (
    <AbsoluteFill style={{ backgroundColor: "white" }}>
      <AbsoluteFill>
        <OffthreadVideo
          style={{
            objectFit: "cover",
          }}
          src={videoSrc}
        />
      </AbsoluteFill>
      {pages.map((page, index) => {
        const nextPage = pages[index + 1] ?? null;
        const subtitleStartFrame = (page.startMs / 1000) * fps;
        const subtitleEndFrame = Math.min(
          nextPage ? (nextPage.startMs / 1000) * fps : Infinity,
          subtitleStartFrame + SWITCH_CAPTIONS_EVERY_MS,
        );
        const durationInFrames = subtitleEndFrame - subtitleStartFrame;
        if (durationInFrames <= 0) {
          return null;
        }

        return (
          <Sequence
            key={index}
            from={subtitleStartFrame}
            durationInFrames={durationInFrames}
          >
            <MinimalStyle key={index} page={page} enterProgress={1} />
          </Sequence>
        );
      })}
      {getFileExists(subtitlesFile) ? null : <NoCaptionFile />}
    </AbsoluteFill>
  );
};