
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "place_result.h"
#include "move.h"
#include "exceptions.h"
#include "human_player.h"
#include "tile_kind.h"
#include "formatting.h"
#include "rang.h"

using namespace std;


// This method is fully implemented.
inline string& to_upper(string& str) {
    transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

inline string& to_lower(string& str) {
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

Move HumanPlayer::get_move(const Board& board, const Dictionary& dictionary) const {
	// TODO: begin your implementation here.
    int count = 0;
    bool repeat = true;
    while(repeat) {
        try {
            string line;
            cout << MOVE_PROMPT_COLOR << "Your move, " << rang::style::reset << PLAYER_NAME_COLOR << this->get_name()  << rang::style::reset << ": ";
            if(board.get_move_index() == 0 && count == 0) {
                cin.ignore(1000, '\n');
            }
            getline(cin, line);
            Move move(parse_move(line));
            move.row--; //Player will enter in 1-1 indexing while computer handles it in 0-0 indexing
            move.column--;
            if(move.tiles.size() > this->count_tiles()) { //number of tiles to place cannot exceed number of tiles
                throw CommandException("Player does not have this many tiles!");
            }
            if(move.kind == MoveKind::PLACE) {
                PlaceResult result = board.test_place(move);
                if(result.valid) {
                    for(size_t i = 0; i < result.words.size(); i++) {
                        if(dictionary.is_word(result.words[i]) == false) {
                            throw MoveException("Forms invalid words!");
                        }
                    }
                } else {
                    throw CommandException(result.error);
                }
                
            } else if(move.kind == MoveKind::EXCHANGE) {
                 Move move(parse_move(line));
            } else {
                 Move move(parse_move(line));
            }
            repeat = false;
            return move;
        } catch(CommandException& e) {
            cerr << e.what() << endl;
            count++;
        } catch(MoveException& e) {
            cerr << e.what() << endl;
            count++;
        }
    }
    Move move;
    return move;
}

vector<TileKind> HumanPlayer::parse_tiles(string& letters) const{
    // TODO: begin implementation here.
    TileCollection temp;
    vector<TileKind> parsed;
    size_t i = 0;
    while(i < letters.size()) {
        try {
            TileKind tile(tiles.lookup_tile(letters[i]));
            temp.add_tile(tile);
            parsed.push_back(tiles.lookup_tile(letters[i]));
        } catch (out_of_range& e) {
            throw CommandException("Error in move: not enough tiles");
        }
        i++;
    }
   
    for(TileCollection::const_iterator it = temp.cbegin(); it != temp.cend(); it++) {
        if(temp.count_tiles(*it) > tiles.count_tiles(*it)) {
            throw CommandException("Error in move: not enough tiles");
        }
    }
    return parsed;
}

Move HumanPlayer::parse_move(string& move_string) const {
	// TODO: begin implementation here.
    stringstream ss(move_string);
    string kind;
    ss >> kind;
    if(to_upper(kind) == "PLACE") {
        char dir;
        size_t row;
        size_t column;
        string letters;
        ss >> dir >> row >> column >> letters;
        letters = to_lower(letters);
        vector<TileKind> parsed;
        try {
            TileCollection temp;
            size_t i = 0;
            string toString = "";
            for(size_t i = 0; i < letters.size(); i++) {
                if(letters[i] == '?') {
                    if(i + 1 < letters.size()) {
                        if(letters[i + 1] == '?') {
                            throw CommandException("No two consecutive blank tiles!");
                        }
                    }
                    if(i + 1 == letters.size()) {
                        throw CommandException("Blank Tile must be followed by a letter!");
                    }
                } else {
                    toString += letters[i];
                }
            }
            if(toString.size() == 0) {
                throw CommandException("No input entered!");
            }
            while(i < letters.size()) {
                try {
                    if(letters[i] == '?') {
                        TileKind tile(tiles.lookup_tile(letters[i]));
                        tile.assigned = letters[i + 1];
                        temp.add_tile(tile);
                        parsed.push_back(tile);
                        i++;
                    } else {
                        TileKind tile(tiles.lookup_tile(letters[i]));
                        temp.add_tile(tile);
                        parsed.push_back(tile);
                    }
                } catch (out_of_range& e) {
                    throw CommandException("Error in move: not enough tiles");
                }
                i++;
            }
            for(TileCollection::const_iterator it = temp.cbegin(); it != temp.cend(); it++) {
                if(temp.count_tiles(*it) > tiles.count_tiles(*it)) {
                    throw CommandException("Error in move: not enough tiles");
                }
            }
        } catch(CommandException& e) {
            throw CommandException(e.what());
        }
        if(dir == '-') {
            Move move(parsed, row, column, Direction::ACROSS);
            return move;
        } else if(dir == '|') {
            Move move(parsed, row, column, Direction::DOWN);
            return move;
        } 
    } else if(to_upper(kind) == "PASS") {
        Move move;
        return move;
    } else if(to_upper(kind) == "EXCHANGE") {
        string letters;
        ss >> letters;
        letters = to_lower(letters);
        vector<TileKind> parsed;
        try {
            parsed = parse_tiles(letters);
            if(parsed.size() == 0) {
                throw CommandException("Enter one or more tiles to exchange!");
            }
        } catch(CommandException& e) {
            throw CommandException(e.what());
        }
        Move move(parsed);
        return move;
    } else {
        throw CommandException("Not a valid Command!");
    }
    Move move;
    return move;
}

// This function is fully implemented.
void HumanPlayer::print_hand(ostream& out) const {
    
    const size_t tile_count = tiles.count_tiles();
    const size_t empty_tile_count = this->get_hand_size() - tile_count;
    const size_t empty_tile_width = empty_tile_count * (SQUARE_OUTER_WIDTH - 1);

    for(size_t i = 0; i < HAND_TOP_MARGIN - 2; ++i) {
        out << endl;
    }

    out << repeat(SPACE, HAND_LEFT_MARGIN) << FG_COLOR_HEADING << "Your Hand: " << endl << endl;

    // Draw top line
    out << repeat(SPACE, HAND_LEFT_MARGIN) << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE;
    print_horizontal(tile_count, L_TOP_LEFT, T_DOWN, L_TOP_RIGHT, out);
    out << repeat(SPACE, empty_tile_width) << BG_COLOR_OUTSIDE_BOARD << endl;

    // Draw middle 3 lines
    for (size_t line = 0; line < SQUARE_INNER_HEIGHT; ++line) {
        out << FG_COLOR_LABEL << BG_COLOR_OUTSIDE_BOARD << repeat(SPACE, HAND_LEFT_MARGIN);
        for (auto it = tiles.cbegin(); it != tiles.cend(); ++it) {
            out << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE << I_VERTICAL << BG_COLOR_PLAYER_HAND;

            // Print letter
            if (line == 1) {
                out << repeat(SPACE, 2) << FG_COLOR_LETTER << (char)toupper(it->letter) << repeat(SPACE, 2);

            // Print score in bottom right
            } else if (line == SQUARE_INNER_HEIGHT - 1) {
                out << FG_COLOR_SCORE << repeat(SPACE, SQUARE_INNER_WIDTH - 2) << setw(2) << it->points;

            } else {
                out << repeat(SPACE, SQUARE_INNER_WIDTH);
            }
        }
        if (tiles.count_tiles() > 0) {
            out << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE << I_VERTICAL;
            out << repeat(SPACE, empty_tile_width) << BG_COLOR_OUTSIDE_BOARD << endl;
        }
    }

    // Draw bottom line
    out << repeat(SPACE, HAND_LEFT_MARGIN) << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE;
    print_horizontal(tile_count, L_BOTTOM_LEFT, T_UP, L_BOTTOM_RIGHT, out);
    out << repeat(SPACE, empty_tile_width) << rang::style::reset << endl;
    out << endl;
}


