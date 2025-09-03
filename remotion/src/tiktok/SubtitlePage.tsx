import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import type {TikTokPage} from '@remotion/captions';
import {Page} from './page';

export const SubtitlePage: React.FC<{readonly page: TikTokPage; readonly highlightColor: string; readonly style: 'neon' | 'boxed' | 'pop' | 'karaoke'}> = ({page, highlightColor, style}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const enter = spring({
    frame,
    fps,
    config: {damping: 200},
    durationInFrames: 5,
  });

  return (
    <AbsoluteFill>
      <Page enterProgress={enter} page={page} highlightColor={highlightColor} style={style} />
    </AbsoluteFill>
  );
};
