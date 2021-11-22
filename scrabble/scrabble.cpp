#include "scrabble.h"
#include "formatting.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>

using namespace std;


// Given to you. this does not need to be changed
Scrabble::Scrabble(const ScrabbleConfig& config)
    : hand_size(config.hand_size)
    , minimum_word_length(config.minimum_word_length)
    , tile_bag(TileBag::read(config.tile_bag_file_path, config.seed))
    , board(Board::read(config.board_file_path))
    , dictionary(Dictionary::read(config.dictionary_file_path))
{ }


void Scrabble::add_players() {
    num_human_players = 0;
    cout << "Please enter number of players: ";
    size_t n;
    cin >> n;
    if(cin.fail()) {
        throw FileException("Not a valid number");
    }
    if(n > 8 || n <= 0) {
        throw FileException("Not a valid number");
    }
    cout << n << " players confirmed." << endl;
    for(size_t i = 0; i < n; i++) {
        cout << "Please enter name for player " << i + 1 << ": ";
        string name;
        cin.ignore(1000, '\n');
        getline(cin, name, '\n');
        cout << "Player " << i + 1 << ", named " << name << " has been added." << endl;
        cout << "Is the player a computer? ";
        char s;
        cin >> s;
        if(s == 'y') {
            players.push_back(make_shared<ComputerPlayer>(name, hand_size));
        } else {
            num_human_players++;
            players.push_back(make_shared<HumanPlayer>(name, hand_size));
        }
        vector<TileKind> moretiles = tile_bag.remove_random_tiles(hand_size); //have to remove tiles based on hand size
        players[i]->add_tiles(moretiles); //add to each player's hand
    }
}

// Game Loop should cycle through players and get and execute that players move
// until the game is over.
void Scrabble::game_loop() {
    // TODO: implement this.
    size_t passCount = 0;
    size_t i = 0;
    bool gameOver = false;
    while(!gameOver) {
        bool bonus = false;
        board.print(cout);
        if(players[i]->is_human()) {
            players[i]->print_hand(cout); //Do not need to print hand for computer player
        }
        
        Move move(players[i]->get_move(board, dictionary));
        if(move.kind == MoveKind::PLACE) {
            size_t points = 0;
            PlaceResult result = board.place(move);
            players[i]->add_points(result.points);
            points += result.points;
            if(move.tiles.size() == hand_size) { //Hand bonus
                players[i]->add_points(50);
                bonus = true;
            }
            
            players[i]->remove_tiles(move.tiles); //We pass the toremove vector to the human player class's remove_tiles() function
            if(players[i]->count_tiles() == 0) {
                if(tile_bag.count_tiles() == 0) {
                    cout << "You gained " << SCORE_COLOR << points << rang::style::reset << " points!" << endl;
                    if(bonus) {
                        cout << "Full hand used! +" << SCORE_COLOR << "50" << rang::style::reset << " points!" << endl;
                        points += 50;
                    }
                    cout << "Your current score: " << SCORE_COLOR << players[i]->get_points() << rang::style::reset << endl;
                    cout << endl << "Press [enter] to continue.";
                    cin.ignore(1000, '\n');
                    break;
                }
            }
            vector<TileKind> moretiles;
            if(move.tiles.size() > tile_bag.count_tiles()) {
                moretiles = tile_bag.remove_random_tiles(tile_bag.count_tiles()); 
            } else {
                moretiles = tile_bag.remove_random_tiles(move.tiles.size());
            }
            players[i]->add_tiles(moretiles);
            cout << "You gained " << SCORE_COLOR << points << rang::style::reset << " points!" << endl;
            if(bonus) {
                cout << "Full hand used! +" << SCORE_COLOR << "50" << rang::style::reset << " points!" << endl;
                points += 50;
            }
            cout << "Your current score: " << SCORE_COLOR << players[i]->get_points() << rang::style::reset << endl;
            cout << endl << "Press [enter] to continue.";
            cin.ignore(100, '\n');
            if(players[i]->is_human()) {
                passCount = 0;
            }
        } else if(move.kind == MoveKind::EXCHANGE) {
            players[i]->remove_tiles(move.tiles); //Remove tiles
            for(size_t i = 0; i < move.tiles.size(); i++) {
                tile_bag.add_tile(move.tiles[i]); //Then add back to the bag
            }
            vector<TileKind> moretiles = tile_bag.remove_random_tiles(move.tiles.size()); //Remove randomly
            players[i]->add_tiles(moretiles);
            cout << "Your current score: " << SCORE_COLOR << players[i]->get_points() << rang::style::reset << endl;
            cout << endl << "Press [enter] to continue.";
            cin.ignore(100, '\n');
            passCount = 0;
        } else {
            cout << "Your current score: " << SCORE_COLOR << players[i]->get_points() << rang::style::reset << endl;
            cout << endl << "Press [enter] to continue.";
            cin.ignore(100, '\n');
            if(players[i]->is_human()) {
                passCount++;
            }
        }
        i++;
        if(i == players.size()) {
            i = 0;
        }
        if(passCount == num_human_players) {
            break;
        }
    }
    // Useful cout expressions with fancy colors. Expressions in curly braces, indicate values you supply.
    // cout << "You gained " << SCORE_COLOR << {points} << rang::style::reset << " points!" << endl;
    // cout << "Your current score: " << SCORE_COLOR << {points} << rang::style::reset << endl;
    // cout << endl << "Press [enter] to continue.";
}

// Performs final score subtraction. Players lose points for each tile in their
// hand. The player who cleared their hand receives all the points lost by the
// other players.
void Scrabble::final_subtraction(vector<shared_ptr<Player>> & players) {
    // TODO: implement this method.
    // Do not change the method signature.
    unsigned int j = 0; //This will be the bonus for any player that clears his hand
    for(size_t i = 0; i < players.size(); i++) {
        if(players[i]->count_tiles() != 0) {
            if(players[i]->get_points() < players[i]->get_hand_value()) {
                j += players[i]->get_hand_value();
                players[i]->subtract_points(players[i]->get_points());
            } else {
                j += players[i]->get_hand_value();
                players[i]->subtract_points(players[i]->get_hand_value());
            }
        }
    }
    for(size_t i = 0; i < players.size(); i++) {
        if(players[i]->count_tiles() == 0) {
            players[i]->add_points(j);
            break;
        }
    }
}

// You should not need to change this function.
void Scrabble::print_result() {
	// Determine highest score
    size_t max_points = 0;
	for (auto player : this->players) {
		if (player->get_points() > max_points) {
			max_points = player->get_points();
        }
	}

	// Determine the winner(s) indexes
	vector<shared_ptr<Player>> winners;
	for (auto player : this->players) {
		if (player->get_points() >= max_points) {
			winners.push_back(player);
        }
	}

    cout << (winners.size() == 1 ? "Winner:" : "Winners: ");
	for (auto player : winners) {
		cout << SPACE << PLAYER_NAME_COLOR << player->get_name();
	}
	cout << rang::style::reset << endl;

	// now print score table
    cout << "Scores: " << endl;
    cout << "---------------------------------" << endl;

	// Justify all integers printed to have the same amount of character as the high score, left-padding with spaces
    cout << setw(static_cast<uint32_t>(floor(log10(max_points) + 1)));

	for (auto player : this->players) {
		cout << SCORE_COLOR << player->get_points() << rang::style::reset << " | " << PLAYER_NAME_COLOR << player->get_name() << rang::style::reset << endl;
	}
}

// You should not need to change this.
void Scrabble::main() {
    add_players();
    game_loop();
    final_subtraction(this->players);
    print_result();
}
