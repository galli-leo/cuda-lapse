#pragma once

/**
 * \brief Represents a 2D point in space.
 */
struct point
{
	int x;
	int y;
};

/**
 * \brief Represents the size of something in 2D.
 */
struct frame_size
{
	unsigned int width;
	unsigned int height;
};

/**
 * \brief Represents the rectangle used for render calculations of something.
 * Idea stolen from UIKit.
 */
struct frame
{
	point pos;
	frame_size size;
};

#pragma region Point Operators

inline bool operator<(const point& p, const point& q)
{
	return p.x < q.x && p.y < q.y;
}

inline bool operator>(const point& p, const point& q)
{
	return q < p;
}

inline bool operator<=(const point& p, const point& q)
{
	return !(p > q);
}

inline bool operator>=(const point& p, const point& q)
{
	return !(p < q);
}

#pragma endregion Point Operators

/**
 * \brief Translate position of frame, such that it is anchored at anchor in relative coordinates.
 * For example, anchor (0, 0) would be set position based on top left corner, (1, 1) based on bottom right corner.
 *
 * Basically we calculate: frame.pos = frame.pos - frame.size * anchor;
 * \param frame
 * \param anchor_y 
 * \param anchor_x
 */
inline void anchor(frame* frame, float anchor_x, float anchor_y)
{
	const int dx = frame->size.width * anchor_x;
	const int dy = frame->size.height * anchor_y;
	frame->pos.x -= dx;
	frame->pos.y -= dy;
}

inline point top_left(frame frame)
{
	return frame.pos;
}

/**
 * \brief Returns bottom right point of the given frame.
 * \param frame 
 * \return 
 */
inline point bottom_right(frame frame)
{
	return point{
		static_cast<int>(frame.size.width) + frame.pos.x,
		static_cast<int>(frame.size.height) + frame.pos.y
	};
}

/**
 * \brief Returns whether point \p p is inside frame \p frame.
 * \param frame 
 * \param p 
 * \return 
 */
inline bool is_inside(frame frame, point p)
{
	return top_left(frame) <= p && p <= bottom_right(frame);
}

/**
 * \brief Returns whether point (\p x, \p y) is inside frame \p frame.
 * \param frame 
 * \param x 
 * \param y 
 * \return 
 */
inline bool is_inside(frame frame, int x, int y)
{
	point p;
	p.x = x;
	p.y = y;
	return is_inside(frame, p);
}