import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, CalculateMetadataFunction, OffthreadVideo, Sequence, delayRender, continueRender, cancelRender, staticFile, watchStaticFile, useVideoConfig} from 'remotion';
import {z} from 'zod';
import type {Caption} from '@remotion/captions';
import {createTikTokStyleCaptions} from '@remotion/captions';
import {getVideoMetadata} from '@remotion/media-utils';
import {SubtitlePage} from './SubtitlePage';

export const captionedVideoSchema = z.object({
  src: z.string(),
  switchEveryMs: z.number().int().min(100).default(1200),
  highlightColor: z.string().default('#39E508'),
  style: z.enum(['neon', 'boxed', 'pop', 'karaoke']).default('neon'),
});

export type CaptionedVideoProps = z.infer<typeof captionedVideoSchema>;

export const calculateCaptionedVideoMetadata: CalculateMetadataFunction<CaptionedVideoProps> = async ({props}) => {
  const fps = 30;
  const metadata = await getVideoMetadata(staticFile(props.src));
  return {
    fps,
    durationInFrames: Math.floor(metadata.durationInSeconds * fps),
  };
};

export const CaptionedVideo: React.FC<CaptionedVideoProps> = ({src, switchEveryMs, highlightColor, style}) => {
  const [subtitles, setSubtitles] = useState<Caption[]>([]);
  const [handle] = useState(() => delayRender());
  const {fps} = useVideoConfig();

  const videoUrl = staticFile(src);
  const subtitlesUrl = videoUrl.replace(/\.(mp4|mkv|mov|webm)$/i, '.json');

  const fetchSubtitles = useCallback(async () => {
    try {
      const res = await fetch(subtitlesUrl);
      const data = (await res.json()) as Caption[];
      setSubtitles(data);
      continueRender(handle);
    } catch (e) {
      cancelRender(e);
    }
  }, [handle, subtitlesUrl]);

  useEffect(() => {
    fetchSubtitles();
    const c = watchStaticFile(subtitlesUrl, () => fetchSubtitles());
    return () => c.cancel();
  }, [fetchSubtitles, subtitlesUrl]);

  const {pages} = useMemo(() => {
    return createTikTokStyleCaptions({
      combineTokensWithinMilliseconds: switchEveryMs,
      captions: subtitles ?? [],
    });
  }, [subtitles, switchEveryMs]);

  return (
    <AbsoluteFill style={{backgroundColor: 'white'}}>
      <AbsoluteFill>
        <OffthreadVideo style={{objectFit: 'cover'}} src={videoUrl} />
      </AbsoluteFill>
      {pages.map((page, index) => {
        const nextPage = pages[index + 1] ?? null;
        const subtitleStartFrame = (page.startMs / 1000) * fps;
        const subtitleEndFrame = Math.min(
          nextPage ? (nextPage.startMs / 1000) * fps : Infinity,
          subtitleStartFrame + switchEveryMs,
        );
        const durationInFrames = subtitleEndFrame - subtitleStartFrame;
        if (durationInFrames <= 0) return null;
        return (
          <Sequence key={index} from={subtitleStartFrame} durationInFrames={durationInFrames}>
            <SubtitlePage page={page} highlightColor={highlightColor} style={style} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
