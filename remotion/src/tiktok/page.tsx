import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import type {TikTokPage} from '@remotion/captions';

const container: React.CSSProperties = {
  justifyContent: 'center',
  alignItems: 'center',
  top: undefined,
  bottom: 260,
  height: 220,
  padding: 0,
};

const BASE_FONT = 'system-ui, -apple-system, Segoe UI, Roboto, Ubuntu';

const styleNeon = (active: boolean, highlight: string): React.CSSProperties => ({
  color: active ? highlight : '#fff',
  textShadow: active
    ? `0 0 6px ${highlight}, 0 0 16px ${highlight}, 0 0 28px ${highlight}`
    : '0 0 12px rgba(0,0,0,0.7)'
});

const styleBoxedWrap: React.CSSProperties = {
  background: 'rgba(0,0,0,0.55)',
  borderRadius: 16,
  padding: '14px 24px',
  display: 'inline-block',
};

const stylePop = (active: boolean, progress: number, highlight: string): React.CSSProperties => ({
  color: active ? highlight : '#fff',
  transform: `scale(${active ? interpolate(progress, [0, 1], [0.9, 1.08]) : 1})`,
  transition: 'transform 150ms ease-out',
});

const styleKaraoke = (active: boolean, highlight: string): React.CSSProperties => ({
  position: 'relative',
  color: '#fff',
  WebkitTextStroke: '1px #000',
  backgroundImage: active
    ? `linear-gradient(90deg, ${highlight} 0%, ${highlight} 100%)`
    : 'none',
  backgroundClip: active ? 'text' as any : undefined,
  WebkitBackgroundClip: active ? 'text' as any : undefined,
  WebkitTextFillColor: active ? 'transparent' : undefined,
});

export const Page: React.FC<{readonly enterProgress: number; readonly page: TikTokPage; readonly highlightColor: string; readonly style: 'neon' | 'boxed' | 'pop' | 'karaoke'}> = ({enterProgress, page, highlightColor, style}) => {
  const frame = useCurrentFrame();
  const {width, fps} = useVideoConfig();
  const timeInMs = (frame / fps) * 1000;

  const fontSize = 64;

  const wrapStyle: React.CSSProperties = {
    fontFamily: BASE_FONT,
    textTransform: 'uppercase',
    width: width * 0.9,
    textAlign: 'center',
    margin: '0 auto',
    fontSize,
    lineHeight: 1.12,
    letterSpacing: 1.2,
    WebkitTextStroke: style === 'neon' || style === 'pop' ? '2px #000' : undefined,
    filter: style === 'neon' ? 'contrast(115%) saturate(110%)' : undefined,
    transform: `scale(${interpolate(enterProgress, [0, 1], [0.96, 1])}) translateY(${interpolate(enterProgress, [0, 1], [40, 0])}px)`,
  };

  return (
    <AbsoluteFill style={container}>
      <div style={style === 'boxed' ? {...styleBoxedWrap, ...wrapStyle} : wrapStyle}>
        {page.tokens.map((t) => {
          const startRelativeToSequence = t.fromMs - page.startMs;
          const endRelativeToSequence = t.toMs - page.startMs;
          const active = startRelativeToSequence <= timeInMs && endRelativeToSequence > timeInMs;
          const tokenProgress = Math.min(1, Math.max(0, (timeInMs - startRelativeToSequence) / Math.max(1, (endRelativeToSequence - startRelativeToSequence))));

          let tokenStyle: React.CSSProperties = {display: 'inline', whiteSpace: 'pre', padding: '0 4px'};

          if (style === 'neon') {
            tokenStyle = {...tokenStyle, ...styleNeon(active, highlightColor)};
          } else if (style === 'boxed') {
            tokenStyle = {
              ...tokenStyle,
              color: active ? '#000' : '#fff',
              background: active ? highlightColor : 'rgba(255,255,255,0.08)',
              borderRadius: 8,
              margin: '0 2px',
            };
          } else if (style === 'pop') {
            tokenStyle = {
              ...tokenStyle,
              ...stylePop(active, tokenProgress, highlightColor),
              textShadow: active ? '0 10px 20px rgba(0,0,0,0.35)' : '0 2px 4px rgba(0,0,0,0.4)'
            };
          } else if (style === 'karaoke') {
            tokenStyle = {
              ...tokenStyle,
              ...styleKaraoke(active, highlightColor),
            };
          }

          return (
            <span key={t.fromMs} style={tokenStyle}>
              {t.text}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
