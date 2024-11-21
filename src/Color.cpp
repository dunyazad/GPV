#include <Color.h>

Color3::Color3(unsigned char r, unsigned char g, unsigned char b)
{
	components[0] = r;
	components[1] = g;
	components[2] = b;
}

void Color3::FromNormalzed(float r, float g, float b)
{
	components[0] = (unsigned char)(r * 255.0f);
	components[1] = (unsigned char)(g * 255.0f);
	components[2] = (unsigned char)(b * 255.0f);
}

const Color3 Color3::Black(0, 0, 0);
const Color3 Color3::Red(255, 0, 0);
const Color3 Color3::Green(0, 255, 0);
const Color3 Color3::Blue(0, 0, 255);
const Color3 Color3::Yellow(255, 255, 0);   // Red + Green
const Color3 Color3::Cyan(0, 255, 255);     // Green + Blue
const Color3 Color3::Magenta(255, 0, 255);  // Red + Blue
const Color3 Color3::White(255, 255, 255);

Color3 Color3::Random()
{
	static std::random_device rd;  // Seed for randomness
	static std::mt19937 gen(rd()); // Mersenne Twister random number engine
	std::uniform_int_distribution<> dis(0, 255); // Range [0, 255]

	return Color3(dis(gen), dis(gen), dis(gen)); // Random r, g, b values
}

Color4::Color4(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	components[0] = r;
	components[1] = g;
	components[2] = b;
	components[3] = a;
}

void Color4::FromNormalzed(float r, float g, float b, float a)
{
	components[0] = (unsigned char)(r * 255.0f);
	components[1] = (unsigned char)(g * 255.0f);
	components[2] = (unsigned char)(b * 255.0f);
	components[3] = (unsigned char)(a * 255.0f);
}

const Color4 Color4::Black(0, 0, 0, 255);
const Color4 Color4::Red(255, 0, 0, 255);
const Color4 Color4::Green(0, 255, 0, 255);
const Color4 Color4::Blue(0, 0, 255, 255);
const Color4 Color4::Yellow(255, 255, 0, 255);   // Red + Green
const Color4 Color4::Cyan(0, 255, 255, 255);     // Green + Blue
const Color4 Color4::Magenta(255, 0, 255, 255);  // Red + Blue
const Color4 Color4::White(255, 255, 255, 255);

Color4 Color4::Random()
{
	static std::random_device rd;  // Seed for randomness
	static std::mt19937 gen(rd()); // Mersenne Twister random number engine
	std::uniform_int_distribution<> dis(0, 255); // Range [0, 255]

	return Color4(dis(gen), dis(gen), dis(gen), dis(gen)); // Random r, g, b, a values
}
