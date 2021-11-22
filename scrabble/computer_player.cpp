
#include <memory>

#include "computer_player.h"
#include <string>
#include <map>
#include <stack>

void ComputerPlayer::left_part(Board::Position anchor_pos, 
                               std::string partial_word,
                               Move partial_move, 
                               std::shared_ptr<Dictionary::TrieNode> node, 
                               size_t limit,
                               TileCollection& remaining_tiles,
                               std::vector<Move>& legal_moves,
                               const Board& board) const {
    // HW5: IMPLEMENT THIS
    if(limit > 0) {
        std::map<char, std::shared_ptr<Dictionary::TrieNode>>::iterator it = node->nexts.begin();
        for(it; it != node->nexts.end(); it++) {
            try {
                TileKind tile = remaining_tiles.lookup_tile(it->first);
                remaining_tiles.remove_tile(tile);
                partial_move.tiles.push_back(tile);
                std::shared_ptr<Dictionary::TrieNode> child = it->second;
                left_part(anchor_pos, partial_word + tile.letter, partial_move, child, limit - 1, remaining_tiles,
                          legal_moves, board); //Recurse if we have the letter in the nexts map
                remaining_tiles.add_tile(tile);
                partial_move.tiles.pop_back();
            } catch(std::out_of_range& e) {
                try {
                    TileKind tile = remaining_tiles.lookup_tile('?'); //If we fail looking up a tile in our present collection, check if we have a blank tile
                    remaining_tiles.remove_tile(tile);
                    tile.assigned = it->first;
                    partial_move.tiles.push_back(tile);
                    std::shared_ptr<Dictionary::TrieNode> child = it->second;
                    left_part(anchor_pos, partial_word + tile.assigned, partial_move, child, limit - 1, remaining_tiles,
                              legal_moves, board);
                    remaining_tiles.add_tile(tile);
                    partial_move.tiles.pop_back();
                } catch(std::out_of_range& e) {
                    continue;
                }
            }
        }
    } else { //Once the limit reaches 0, we can extend right the prefix we have so far
        extend_right(anchor_pos, partial_word, partial_move, node, remaining_tiles, legal_moves, board);
    }
}

void ComputerPlayer::extend_right(Board::Position square,
                                  std::string partial_word,
                                  Move partial_move, 
                                  std::shared_ptr<Dictionary::TrieNode> node,
                                  TileCollection& remaining_tiles,
                                  std::vector<Move>& legal_moves,
                                  const Board& board) const {
    // HW5: IMPLEMENT THIS
    if(board.is_in_bounds(square)) {
        if(!(board.in_bounds_and_has_tile(square))) {
                std::map<char, std::shared_ptr<Dictionary::TrieNode>>::iterator it = node->nexts.begin();
                for(it; it != node->nexts.end(); it++) {
                    try {
                        TileKind tile = remaining_tiles.lookup_tile(it->first);
                        remaining_tiles.remove_tile(tile);
                        partial_move.tiles.push_back(tile);
                        std::shared_ptr<Dictionary::TrieNode> child = it->second;
                        if(child->is_final) {
                            legal_moves.push_back(partial_move); //Push back the partial word whenever encountering a valid word
                        }
                        Board::Position other = square.translate(partial_move.direction);
                        extend_right(other, partial_word + tile.letter, partial_move, child, remaining_tiles,
                                  legal_moves, board);
                        partial_move.tiles.pop_back();
                        remaining_tiles.add_tile(tile);
                    } catch(std::out_of_range& e) {
                        try {
                            TileKind tile = remaining_tiles.lookup_tile('?');
                            remaining_tiles.remove_tile(tile);
                            tile.assigned = it->first;
                            partial_move.tiles.push_back(tile);
                            std::shared_ptr<Dictionary::TrieNode> child = it->second;
                            if(child->is_final) {
                                legal_moves.push_back(partial_move);
                            }
                            Board::Position other = square.translate(partial_move.direction);
                            extend_right(other, partial_word + tile.assigned, partial_move, child, remaining_tiles,
                                      legal_moves, board);
                            partial_move.tiles.pop_back();
                            remaining_tiles.add_tile(tile);
                        } catch(std::out_of_range& e) {
                            continue;
                        }
                    }
                }
            } else { //If there is already a letter at the position, we have to see if it exists in the current node's nexts map
                std::map<char, std::shared_ptr<Dictionary::TrieNode>>::iterator it = node->nexts.find(board.letter_at(square));
                if(it ==
                   node->nexts.end()) {
                    return;
                } else {
                    std::shared_ptr<Dictionary::TrieNode> child = it->second;
                    if(child->is_final) {
                        legal_moves.push_back(partial_move); //If it is, push back to legal_moves
                    }
                    extend_right(square.translate(partial_move.direction), partial_word + board.letter_at(square), partial_move, child, remaining_tiles,
                              legal_moves, board);
                }
            }
    } else {
        return;
    }
}

Move ComputerPlayer::get_move(const Board& board, const Dictionary& dictionary) const {
	std::vector<Move> legal_moves;
    std::vector<Board::Anchor> anchors = board.get_anchors();
    // HW5: IMPLEMENT THIS
    TileCollection tempCollection(this->tiles);
    for(Board::Anchor anchor : anchors) {
        if(anchor.limit == 0) { //If the limit is zero, we have to check for all the previous letters in the anchor's direction that will be part of the prefix
            std::string preword = "";
            Board::Position temp = anchor.position.translate(anchor.direction, -1);
            std::stack<char> pre;
            while(board.in_bounds_and_has_tile(temp)) {
                pre.push(board.letter_at(temp));
                temp = temp.translate(anchor.direction, -1);
            }
            while(!pre.empty()) {
                preword += pre.top();
                pre.pop();
            }
            Move move;
            move.direction = anchor.direction;
            move.row = anchor.position.row;
            move.column = anchor.position.column;
            std::shared_ptr<Dictionary::TrieNode> node = dictionary.find_prefix(preword);
            
            extend_right(anchor.position, preword, move, node, tempCollection, legal_moves, board);
        } else {
            Move move;
            move.direction = anchor.direction;
            for(size_t i = 0; i <= anchor.limit; i++) { //Call left_part with limits from 1 up to the anchor.limit
                if(anchor.direction == Direction::ACROSS) { //Set the row and column of the move
                    move.row = anchor.position.row;
                    move.column = anchor.position.column - i;
                } else {
                    move.row = anchor.position.row - i;
                    move.column = anchor.position.column;
                }
                left_part(anchor.position, "", move, dictionary.get_root(), i, tempCollection, legal_moves, board);
            }
        }
    }
	
	return get_best_move(legal_moves, board, dictionary);
}

Move ComputerPlayer::get_best_move(std::vector<Move> legal_moves, const Board& board, const Dictionary& dictionary) const {
    Move best_move = Move(); // Pass if no move found
	// HW5: IMPLEMENT THIS
    size_t maxPoints = 0;
    size_t index = 0;
    bool moveFound = false;
    for(size_t i = 0; i < legal_moves.size(); i++) {
        PlaceResult result = board.test_place(legal_moves[i]);
        if(result.valid == false) {
            continue;
        }
        bool validWords = true;
        for(size_t i = 0; i < result.words.size(); i++) {
            if(!dictionary.is_word(result.words[i])) { //Every single word in the words vector of PlaceResult has to be valid (we haven't checked perpendicular words yet!)
                validWords = false;
                break;
            }
        }
        if(!validWords) {
            continue;
        }
            
        moveFound = true;
        size_t temp = 0;
        if(legal_moves[i].tiles.size() == this->get_hand_size()) {
            temp = 50; //Account for a move that uses the full hand size
        }
        if(result.points + temp > maxPoints) {
            index = i;
            maxPoints = result.points + temp;
        }
    }
    if(!moveFound) {
        return best_move;
    }
    
    best_move = legal_moves[index];
    best_move.kind = MoveKind::PLACE;
	return best_move;	
}
