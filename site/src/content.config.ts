import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";
import { defineCollection } from "astro:content";
import { z } from "astro/zod";

// Every page has a title, a one-line description and one Diátaxis type. The build fails without
// them, and `npm run check` checks that each page sits in the folder of its type.
export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        description: z.string().min(20),
        type: z.enum(["tutorial", "how-to", "reference", "explanation", "hub", "project"]),
      }),
    }),
  }),
};
