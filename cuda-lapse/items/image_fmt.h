#pragma once
#include "image.h"

#pragma warning(push, 0)        
#include "spdlog/fmt/fmt.h"
#pragma warning(pop)

namespace fmt {
	template <>
	struct formatter<image> {
		template <typename ParseContext>
		constexpr auto parse(ParseContext &ctx) { return ctx.begin(); }

		template <typename FormatContext>
		auto format(const image &p, FormatContext &ctx) {
			return format_to(ctx.out(), "Image({}, state: {}, timestamp: {}, path: {})", p.id, state_to_string(p.state), p.timestamp, p.path);
		}
	};
}
