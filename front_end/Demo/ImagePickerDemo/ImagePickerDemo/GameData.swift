//
//  Data.swift
//  ImagePickerDemo
//
//  Created by mac on 2021/10/8.
//  Copyright © 2021 Ramon Gilabert Llop. All rights reserved.
//

import Foundation
class GameData: NSObject {

    static let shared = GameData()
    // Make sure the class has only one instance
    // Should not init or copy outside
    lazy var player1 = [Int]()
    lazy var player2 = [Int]()
    lazy var actionIndex = 0
  
    private override init() {
    }
    
    override func copy() -> Any {
        return self // SingletonClass.shared
    }
    
    override func mutableCopy() -> Any {
        return self // SingletonClass.shared
    }
  func getPlayer1Data() -> [Int] {
    return player1
  }
  
  func getPlayer2Data() -> [Int] {
    return player2
  }
  
  func setActorIndex(_index:Int) {
    actionIndex = _index
  }
  func getActionIndex() -> Int {
    return actionIndex
  }
  func addPlayScore(_score:Int) {
    if actionIndex == 0 {
      addPlayer1(_score: _score)
    }
    else
    {
      addPlayer2(_score: _score)
    }
  }
  
  func player1TotleScore() -> Int {
    var totleScore:Int = 0
    for item in player1 {
      totleScore = totleScore + item
    }
    return totleScore
  }
  
  func player2TotleScore() -> Int {
    var totleScore:Int = 0
    for item in player2 {
      totleScore = totleScore + item
    }
    return totleScore
  }
  
  func addPlayer1(_score:Int) {
      if player1.count < 8 {
        player1.append(_score)
      }
    }
  
  func addPlayer2(_score:Int) {
    if player2.count < 8 {
      player2.append(_score)
    }
  }
  
    // Optional
    func reset() {
        // Reset all properties to default value
        player1 = [Int]()
        player2 = [Int]()
    }
}
