You are a stranger scrolling TikTok. You see this clip cold.

You have never seen this show. You know nothing about the characters, the
story, or what happened before. All you get is the clip below, written out as
a script: lines in [brackets] are what you see on screen, "Speaker A:",
"Speaker B:" ... are different people talking (you see their faces, but nobody
tells you their names unless someone says a name out loud). "[cut]" marks a
jump to a later moment.

Judge only from what is in the clip. Do not guess at backstory. If you cannot
tell who someone is or what they want, say so.

Return the JSON directly without extended reasoning.

Return strict JSON:
{
  "what_is_happening": "<one sentence: what you just watched>",
  "who_are_they": "<one sentence: who these people are to each other, as far as the clip tells you>",
  "open_question": "<one sentence: the question the clip leaves you with, or 'none'>",
  "followable_score": <1-5: 1 = lost, 3 = got the gist, 5 = fully followed who wants what from whom>,
  "would_tap_next": <true if you would tap to see what happens next, else false>
}
