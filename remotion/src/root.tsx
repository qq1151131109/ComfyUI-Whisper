import React from 'react';
import {Composition} from 'remotion';
import {CaptionedVideo, captionedVideoSchema, calculateCaptionedVideoMetadata} from './tiktok/CaptionedVideo';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="CaptionedVideo"
        component={CaptionedVideo}
        calculateMetadata={calculateCaptionedVideoMetadata}
        schema={captionedVideoSchema}
        width={720}
        height={1280}
        defaultProps={{
          src: 'video.mp4',
          switchEveryMs: 1200,
          highlightColor: '#39E508',
          style: 'neon',
        }}
      />
    </>
  );
};
