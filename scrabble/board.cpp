#include "board.h"
#include "board_square.h"
#include "exceptions.h"
#include "formatting.h"
#include <iomanip>
#include <fstream>
#include <math.h>

using namespace std;


bool Board::Position::operator==(const Board::Position& other) const {
    return this->row == other.row && this->column == other.column;
}

bool Board::Position::operator!=(const Board::Position& other) const {
    return this->row != other.row || this->column != other.column;
}

Board::Position Board::Position::translate(Direction direction) const {
    return this->translate(direction, 1);
}

Board::Position Board::Position::translate(Direction direction, ssize_t distance) const {
    if (direction == Direction::DOWN) {
        return Board::Position(this->row + distance, this->column);
    } else {
        return Board::Position(this->row, this->column + distance);
    }
}

char Board::letter_at(Position p) const {
    if(at(p).get_tile_kind().letter == '?') { //If the tile is a blank tile, return the assigned letter
        return at(p).get_tile_kind().assigned;
    }
    return at(p).get_tile_kind().letter;
}

bool Board::is_anchor_spot(Position p) const {
    if(get_move_index() == 0) {
        if(p != start) {
            return false;
        }
        return true;
    }
    if(in_bounds_and_has_tile(p)) { //Has to be be empty spot and in bounds
        return false;
    }
    if(is_in_bounds(p)) {
        p = p.translate(Direction::ACROSS, -1);
        if(is_in_bounds(p)) {
            if(at(p).has_tile()) { //As long as one of the neighboring positions has a tile, return true
                return true;
            }
        }
        p = p.translate(Direction::ACROSS);
        p = p.translate(Direction::DOWN, -1);
        if(is_in_bounds(p)) {
            if(at(p).has_tile()) {
                return true;
            }
        }
        p = p.translate(Direction::DOWN, 2);
        if(is_in_bounds(p)) {
            if(at(p).has_tile()) {
                return true;
            }
        }
        p = p.translate(Direction::DOWN, -1);
        p = p.translate(Direction::ACROSS);
        
        if(is_in_bounds(p)) {
            if(at(p).has_tile()) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Board::Anchor> Board::get_anchors() const {
    vector<Board::Anchor> anchors;
    if(get_move_index() == 0) {
        Board::Anchor anchor1(start, Direction::ACROSS, start.column);
        anchors.push_back(anchor1); //The start square represents an anchor in each direction, so two
        Board::Anchor anchor2(start, Direction::DOWN, start.row);
        anchors.push_back(anchor2);
        return anchors;
    }
    
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < columns; j++) {
            Position pos(i, j);
            Position temp(i, j);
            //add two anchors for each anchor_spot position, one in down position and one across position
            if(is_anchor_spot(pos)) {
                int acrossCount = 0;
                while(is_in_bounds(temp)) {
                    temp = temp.translate(Direction::ACROSS, -1);
                    if(is_in_bounds(temp)) {
                        if((!is_anchor_spot(temp)) && (!at(temp).has_tile())) {
                            acrossCount++;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                Board::Anchor anchor(pos, Direction::ACROSS, acrossCount);
                anchors.push_back(anchor);
                
                temp.row = i;
                temp.column = j;
                int downCount = 0;
                while(is_in_bounds(temp)) {
                    temp = temp.translate(Direction::DOWN, -1);
                    if(is_in_bounds(temp)) {
                        if(!(is_anchor_spot(temp)) && !(at(temp).has_tile())) {
                            downCount++;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                Board::Anchor anchor2(pos, Direction::DOWN, downCount);
                anchors.push_back(anchor2);
            }
        }
    }
    return anchors;
    
}

Board Board::read(const string& file_path) {
    ifstream file(file_path);
    if (!file) {
        throw FileException("cannot open board file!");
    }

    size_t rows;
    size_t columns;
    size_t starting_row;
    size_t starting_column;
    file >> rows >> columns >> starting_row >> starting_column;
    Board board(rows, columns, starting_row, starting_column);
    
        
    // TODO: complete implementation of reading in board from file here.
    string line;
    for(size_t i = 0; i < rows; i++) {
        file >> line;
        board.squares.push_back({});
        for(size_t j = 0; j < columns; j++) {
            if(line.at(j) == '.') {
                board.squares.at(i).push_back(BoardSquare(1, 1));
            } else if(line.at(j) == 't') {
                board.squares.at(i).push_back(BoardSquare(1, 3));
            } else if(line.at(j) == 'd') {
                board.squares.at(i).push_back(BoardSquare(1, 2));
            } else if(line.at(j) == '2') {
                board.squares.at(i).push_back(BoardSquare(2, 1));
            } else if(line.at(j) == '3') {
                board.squares.at(i).push_back(BoardSquare(3, 1));
            }
        }
    }
    return board;
}

size_t Board::get_move_index() const {
    return this->move_index;
}

PlaceResult Board::test_place(const Move& move) const {
    // TODO: complete implementation here
    Board::Position begin(move.row, move.column);
    if(!(is_in_bounds(begin))) { //Make sure it is a valid index
        PlaceResult notValid("Not a valid index");
        return notValid;
    }

    if(at(begin).has_tile()) { //Cannot square on a square with a tile
        PlaceResult notValid("Cannot start on a square with a tile!");
        return notValid;
    }
    bool alone = true;
    bool attached = false;
    if(get_move_index() == 0 && move.direction == Direction::ACROSS) { //The first move has to encompass the start square
        if(!(begin.column <= start.column && begin.column + move.tiles.size() - 1 >= start.column && begin.row == start.row)) {
            PlaceResult notValid("Not starting on start square!");
            return notValid;
        }
    } else if(get_move_index() == 0 && move.direction == Direction::DOWN) {
        if(!(begin.row <= start.row && begin.row + move.tiles.size() - 1 >= start.row && begin.column == start.column)) {
            PlaceResult notValid("Not starting on start square!");
            return notValid;
        }
    }
    int dir2 = 0; //word-multipliers in direction of placement
    int dir3 = 0;
    vector<TileKind> s;
    unsigned short points = 0;
    vector<string> words;
    size_t i = 0;
    Board::Position pos(0, 0);
    if(move.direction == Direction::ACROSS || move.direction == Direction::DOWN) {
        if(move.direction == Direction::ACROSS) {
            pos.row = move.row;
        } else {
            pos.column = move.column;
        }
        int vpoints = 0;
        int spoints = 0;
        while(is_in_bounds(pos)) { //Nested while loops for horizontal and vertical forming of words
            if(at(pos).has_tile()) {
                s.push_back(at(pos).get_tile_kind());
                spoints += (at(pos).get_tile_kind().points);
                if(i > 0) { //This indicates that we have connected the newly placed tiles with already-placed ones
                    attached = true;
                }
            } else {
                if(move.direction == Direction::ACROSS) {
                    if(pos.column < move.column) {
                        while(!s.empty()) {
                            s.pop_back();
                            spoints = 0;
                        }
                        pos.column++;
                        continue;
                    }
                    if(pos.column == move.column) {
                        if(s.size() > 0) { //If the size of s is greater than 0, the word is attached to a previous word placed before it
                            attached = true;
                        }
                    }
                } else {
                    if(pos.row < move.row) {
                        while(!s.empty()) {
                            s.pop_back();
                            spoints = 0;
                        }
                        pos.row++;
                        continue;
                    }
                    if(pos.row == move.row) {
                        if(s.size() > 0) {
                            attached = true;
                        }
                    }
                }
                if(i >= move.tiles.size()) { //We have completed iterating through the new tiles
                    break;
                }
                int count2 = 0;
                int count3 = 0;
                Board::Position temp(0, 0);
                if(move.direction == Direction::ACROSS) {
                    temp.column = pos.column;
                } else if(move.direction == Direction::DOWN){
                    temp.row = pos.row;
                }
                bool after = false;
                vector<TileKind> v;
                while(is_in_bounds(temp)) { //This while loop checks temp and all the words that are formed through the perpendicular direction.
                    bool movepos;
                    if(move.direction == Direction::ACROSS) {
                        movepos = (temp.row == move.row);
                    } else {
                        movepos = (temp.column == move.column);
                    }
                    if(movepos) {
                            v.push_back(move.tiles[i]);
                            s.push_back(move.tiles[i]);
                            after = true;
                            if(squares[temp.row][temp.column].word_multiplier == 3) { //Here we check for all word multipliers
                                count3 += 1;
                                dir3 += 1;
                            } else if(squares[temp.row][temp.column].word_multiplier == 2) {
                                count2 += 1;
                                dir2 += 1;
                            }
                            vpoints += (move.tiles[i].points * squares[temp.row][temp.column].letter_multiplier);
                            spoints += (move.tiles[i].points * squares[temp.row][temp.column].letter_multiplier);
                        i++;
                    } else if(!at(temp).has_tile()) {
                        if(after) {
                            break;
                        } else {
                            while(!v.empty()) {
                                v.pop_back();
                                vpoints = 0;
                            }
                        }
                    } else {
                        v.push_back(at(temp).get_tile_kind());
                        vpoints += at(temp).get_tile_kind().points;
                    }
                    if(move.direction == Direction::ACROSS) {
                        temp.row++;
                    } else {
                        temp.column++;
                    }
                }
                if(v.size() > 1) {
                    alone = false;
                    string s = "";
                    for(size_t i = 0; i < v.size(); i++) {
                        if(v[i].letter == '?') {
                            s += v[i].assigned;
                        } else {
                            s += v[i].letter;
                        }
                    }
                    if(count3 > 0) {
                        vpoints *= pow(3, count3);
                    }
                    if(count2 > 0) {
                        vpoints *= pow(2, count2);
                    }
                    points += vpoints;
                    words.push_back(s);
                }
                vpoints = 0;
                while(!v.empty()) {
                    v.pop_back();
                }
                count2 = 0;
                count3 = 0;
            }
            if(move.direction == Direction::ACROSS) {
                pos.column++;
            } else {
                pos.row++;
            }
        }
        if(i < move.tiles.size()) { //This is if we aren't able to place the word because it is not in bounds
            PlaceResult notValid("Invalid bounds!");
            return notValid;
        }
        if(alone == true && attached == false) { //It has to be attached horizontally and/or vertically
            if(move_index > 0) {
                PlaceResult notValid("Has to be attached to another word!");
                return notValid;
            }
            if(move.tiles.size() < 2) {
                PlaceResult notValid("Words have to be at least length 2");
                return notValid;
            }
        }
        if(s.size() > 1) {
            string word = "";
            for(size_t i = 0; i < s.size(); i++) {
                if(s[i].letter == '?') {
                    word += s[i].assigned;
                } else {
                    word += s[i].letter;
                }
            }
            if(dir3 > 0) {
                spoints *= pow(3, dir3); //Add 3-multipliers
            }
            if(dir2 > 0) {
                spoints *= pow(2, dir2); //Add 2-multipliers
            }
            points += spoints;
            words.push_back(word);
        }
        PlaceResult result(words, points);
        return result;
    } else {
        PlaceResult result("Not a valid direction!");
        return result;
    }
}



PlaceResult Board::place(const Move& move) {
    // TODO: Complete implementation here
    PlaceResult result = test_place(move);
    if(result.valid) { //Only if the result is valid do we place anything
        if(move.direction == Direction::DOWN) {
            Board::Position pos(move.row, move.column);
            size_t i = 0;
            while(i < move.tiles.size()) { //Simply set the tile
                if(!squares[pos.row][pos.column].has_tile()) {
                    squares[pos.row][pos.column].set_tile_kind(move.tiles[i]);
                    i++;
                }
                pos.row++;
            }
        } else if(move.direction == Direction::ACROSS) {
            Board::Position pos(move.row, move.column);
            size_t i = 0;
            while(i < move.tiles.size()) {
                if(!squares[pos.row][pos.column].has_tile()) {
                    squares[pos.row][pos.column].set_tile_kind(move.tiles[i]);
                    i++;
                }
                pos.column++;
            }
        }
        move_index++;
    }

    return result;
}

// The rest of this file is provided for you. No need to make changes.

BoardSquare& Board::at(const Board::Position& position) {
    return this->squares.at(position.row).at(position.column);
}

const BoardSquare& Board::at(const Board::Position& position) const {
    return this->squares.at(position.row).at(position.column);
}

bool Board::is_in_bounds(const Board::Position& position) const {
    return position.row < this->rows && position.column < this->columns;
}

bool Board::in_bounds_and_has_tile(const Position& position) const{
    return is_in_bounds(position) && at(position).has_tile();
}

void Board::print(ostream& out) const {
	// Draw horizontal number labels
	for (size_t i = 0; i < BOARD_TOP_MARGIN - 2; ++i) {
		out << std::endl;
	}
	out << FG_COLOR_LABEL << repeat(SPACE, BOARD_LEFT_MARGIN);
	const size_t right_number_space = (SQUARE_OUTER_WIDTH - 3) / 2;
	const size_t left_number_space = (SQUARE_OUTER_WIDTH - 3) - right_number_space;
	for (size_t column = 0; column < this->columns; ++column) {
		out << repeat(SPACE, left_number_space) << std::setw(2) << column + 1 << repeat(SPACE, right_number_space);
	}
	out << std::endl;

	// Draw top line
	out << repeat(SPACE, BOARD_LEFT_MARGIN);
    print_horizontal(this->columns, L_TOP_LEFT, T_DOWN, L_TOP_RIGHT, out);
    out << endl;

	// Draw inner board
	for (size_t row = 0; row < this->rows; ++row) {
        if (row > 0) {
            out << repeat(SPACE, BOARD_LEFT_MARGIN);
			print_horizontal(this->columns, T_RIGHT, PLUS, T_LEFT, out);
			out << endl;
		}

		// Draw insides of squares
		for (size_t line = 0; line < SQUARE_INNER_HEIGHT; ++line) {
			out << FG_COLOR_LABEL << BG_COLOR_OUTSIDE_BOARD;

			// Output column number of left padding
			if (line == 1) {
				out << repeat(SPACE, BOARD_LEFT_MARGIN - 3);
				out << std::setw(2) << row + 1;
				out << SPACE;
			} else {
				out << repeat(SPACE, BOARD_LEFT_MARGIN);
			}

            // Iterate columns
			for (size_t column = 0; column < this->columns; ++column) {
				out << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE << I_VERTICAL;
				const BoardSquare& square = this->squares.at(row).at(column);
				bool is_start = this->start.row == row && this->start.column == column;

				// Figure out background color
				if (square.word_multiplier == 2) {
					out << BG_COLOR_WORD_MULTIPLIER_2X;
				} else if (square.word_multiplier == 3) {
					out << BG_COLOR_WORD_MULTIPLIER_3X;
				} else if (square.letter_multiplier == 2) {
					out << BG_COLOR_LETTER_MULTIPLIER_2X;
				} else if (square.letter_multiplier == 3) {
					out << BG_COLOR_LETTER_MULTIPLIER_3X;
				} else if (is_start) {
					out << BG_COLOR_START_SQUARE;
                }

				// Text
                if (line == 0 && is_start) {
                    out << "  \u2605  ";
                } else if (line == 0 && square.word_multiplier > 1) {
                    out << FG_COLOR_MULTIPLIER << repeat(SPACE, SQUARE_INNER_WIDTH - 2) << 'W' << std::setw(1) << square.word_multiplier;
                } else if (line == 0 && square.letter_multiplier > 1) {
                    out << FG_COLOR_MULTIPLIER << repeat(SPACE, SQUARE_INNER_WIDTH - 2) << 'L' << std::setw(1) << square.letter_multiplier;
				} else if (line == 1 && square.has_tile()) {
                    char l = square.get_tile_kind().letter == TileKind::BLANK_LETTER ? square.get_tile_kind().assigned : ' ';
                    out << repeat(SPACE, 2) << FG_COLOR_LETTER << square.get_tile_kind().letter << l << repeat(SPACE, 1);
				} else if (line == SQUARE_INNER_HEIGHT - 1 && square.has_tile()) {
                    if(square.get_tile_kind().points > 9) {
                        out << repeat(SPACE, SQUARE_INNER_WIDTH - 2) << FG_COLOR_SCORE << square.get_tile_kind().points;
                    } else {
                        out << repeat(SPACE, SQUARE_INNER_WIDTH - 1) << FG_COLOR_SCORE << square.get_tile_kind().points;
                    }
				} else {
					out << repeat(SPACE, SQUARE_INNER_WIDTH);
				}
			}

			// Add vertical line
			out << FG_COLOR_LINE << BG_COLOR_NORMAL_SQUARE << I_VERTICAL << BG_COLOR_OUTSIDE_BOARD << std::endl;
		}
	}

	// Draw bottom line
	out << repeat(SPACE, BOARD_LEFT_MARGIN);
	print_horizontal(this->columns, L_BOTTOM_LEFT, T_UP, L_BOTTOM_RIGHT, out);
	out << endl << rang::style::reset << std::endl;
}
