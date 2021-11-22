#include "player.h"

using namespace std;


// TODO: implement member functions

void Player::add_points(size_t points) {
    this->points += points;
}

void Player::subtract_points(size_t points) {
    this->points -= points;
}

size_t Player::get_points() const {
    return points;
}

const string& Player::get_name() const {
    return name;
}

void Player::remove_tiles(const std::vector<TileKind>& tiles) {
    for(size_t i = 0; i < tiles.size(); i++) {
        this->tiles.remove_tile(tiles[i]);
    }
}

size_t Player::count_tiles() const {
    return tiles.count_tiles();
}

void Player::add_tiles(const std::vector<TileKind>& tiles) {
    for(size_t i = 0; i < tiles.size(); i++) {
        this->tiles.add_tile(tiles[i]);
    }
}

bool Player::has_tile(TileKind tile) {
    try {
        tiles.lookup_tile(tile.letter); //The lookup function throws an exception which we catch here
    } catch(out_of_range& e) {
        return false;
    }
    return true;
}

unsigned int Player::get_hand_value() const {
    return tiles.total_points();
}

size_t Player::get_hand_size() const {
    return hand_size;
}
